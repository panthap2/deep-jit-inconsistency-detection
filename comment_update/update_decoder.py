from dpu_utils.mlutils import Vocabulary
import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_add

from constants import START, BEAM_SIZE
from decoder import Decoder
from tensor_utils import compute_attention_states

class UpdateDecoder(Decoder):
    def __init__(self, input_size, hidden_size, attention_state_size, embedding_store,
                 embedding_size, dropout_rate, attn_input_size):
        """Decoder for the edit model which generates a sequence of NL edits based on learned representations of
           the old comment and code edits."""
        super(UpdateDecoder, self).__init__(input_size, hidden_size, attention_state_size,
            embedding_store, embedding_size, dropout_rate)
        
        self.sequence_attention_code_transform_matrix = nn.Parameter(
            torch.randn(self.attention_state_size, self.hidden_size,
                dtype=torch.float, requires_grad=True)
            )
        self.attention_old_nl_hidden_transform_matrix = nn.Parameter(
            torch.randn(self.attention_state_size, self.hidden_size,
                dtype=torch.float, requires_grad=True)
            )

        self.attention_output_layer = nn.Linear(attn_input_size + self.hidden_size,
            self.hidden_size, bias=False)

    def decode(self, initial_state, decoder_input_embeddings, encoder_hidden_states,
               code_hidden_states, old_nl_hidden_states, masks, code_masks, old_nl_masks):
        """Decoding with attention and copy. Attention is computed separately for each set of encoder hidden states."""
        decoder_states, decoder_final_state = self.gru.forward(decoder_input_embeddings, initial_state.unsqueeze(0))
        
        attention_context_states = compute_attention_states(old_nl_hidden_states, old_nl_masks,
            decoder_states, self.attention_old_nl_hidden_transform_matrix, None)

        code_contexts = compute_attention_states(code_hidden_states, code_masks,
            decoder_states, self.sequence_attention_code_transform_matrix, None)
        attention_context_states = torch.cat([attention_context_states, code_contexts], dim=-1)
        
        decoder_states = torch.tanh(self.attention_output_layer(
            torch.cat([attention_context_states, decoder_states], dim=-1)))
        
        generation_scores = torch.einsum('ijk,km->ijm', decoder_states, self.generation_output_matrix)
        copy_scores = torch.einsum('ijk,km,inm->inj', encoder_hidden_states,
            self.copy_encoder_hidden_transform_matrix, decoder_states)
        copy_scores.masked_fill_(masks, float('-inf'))

        combined_logprobs = nn.functional.log_softmax(torch.cat([generation_scores, copy_scores], dim=-1), dim=-1)
        generation_logprobs = combined_logprobs[:,:,:len(self.embedding_store.nl_vocabulary)]
        copy_logprobs = combined_logprobs[:, :,len(self.embedding_store.nl_vocabulary):]
        
        return decoder_states, decoder_final_state, generation_logprobs, copy_logprobs

    def forward(self, initial_state, decoder_input_embeddings, encoder_hidden_states,
                code_hidden_states, old_nl_hidden_states, masks, code_masks, old_nl_masks):
        """Runs decoding."""
        return self.decode(initial_state, decoder_input_embeddings, encoder_hidden_states,
                           code_hidden_states, old_nl_hidden_states, masks, code_masks, old_nl_masks)
    
    def beam_decode(self, initial_state, encoder_hidden_states, code_hidden_states, old_nl_hidden_states,
                    masks, max_out_len, batch_data, code_masks, old_nl_masks, device):
        """Beam search. Generates the top K candidate predictions."""
        batch_size = initial_state.shape[0]
        decoded_batch = [list() for _ in range(batch_size)]
        decoded_batch_scores = np.zeros([batch_size, BEAM_SIZE])

        decoder_input = torch.tensor(
            [[self.embedding_store.get_nl_id(START)]] * batch_size, device=device)
        decoder_input = decoder_input.unsqueeze(1)
        decoder_state = initial_state.unsqueeze(1).expand(
            -1, decoder_input.shape[1], -1).reshape(-1, initial_state.shape[-1])

        beam_scores = torch.ones([batch_size, 1], dtype=torch.float32, device=device)
        beam_status = torch.zeros([batch_size, 1], dtype=torch.uint8, device=device)
        beam_predicted_ids = torch.full([batch_size, 1, max_out_len], self.embedding_store.get_end_id(),
            dtype=torch.int64, device=device)

        for i in range(max_out_len):
            beam_size = decoder_input.shape[1]
            if beam_status[:,0].sum() == batch_size:
                break

            tiled_encoder_states = encoder_hidden_states.unsqueeze(1).expand(-1, beam_size, -1, -1)
            tiled_masks = masks.unsqueeze(1).expand(-1, beam_size, -1, -1)
            tiled_code_hidden_states = code_hidden_states.unsqueeze(1).expand(-1, beam_size, -1, -1)
            tiled_code_masks = code_masks.unsqueeze(1).expand(-1, beam_size, -1, -1)
            tiled_old_nl_hidden_states = old_nl_hidden_states.unsqueeze(1).expand(-1, beam_size, -1, -1)
            tiled_old_nl_masks = old_nl_masks.unsqueeze(1).expand(-1, beam_size, -1, -1)

            flat_decoder_input = decoder_input.reshape(-1, decoder_input.shape[-1])
            flat_encoder_states = tiled_encoder_states.reshape(-1, tiled_encoder_states.shape[-2], tiled_encoder_states.shape[-1])
            flat_masks = tiled_masks.reshape(-1, tiled_masks.shape[-2], tiled_masks.shape[-1])
            flat_code_hidden_states = tiled_code_hidden_states.reshape(-1, tiled_code_hidden_states.shape[-2], tiled_code_hidden_states.shape[-1])
            flat_code_masks = tiled_code_masks.reshape(-1, tiled_code_masks.shape[-2], tiled_code_masks.shape[-1])
            flat_old_nl_hidden_states = tiled_old_nl_hidden_states.reshape(-1, tiled_old_nl_hidden_states.shape[-2], tiled_old_nl_hidden_states.shape[-1])
            flat_old_nl_masks = tiled_old_nl_masks.reshape(-1, tiled_old_nl_masks.shape[-2], tiled_old_nl_masks.shape[-1])

            decoder_input_embeddings = self.embedding_store.get_nl_embeddings(flat_decoder_input)
            decoder_attention_states, flat_decoder_state, generation_logprobs, copy_logprobs = self.decode(
                decoder_state, decoder_input_embeddings, flat_encoder_states, flat_code_hidden_states, 
                flat_old_nl_hidden_states, flat_masks, flat_code_masks, flat_old_nl_masks)
            
            generation_logprobs = generation_logprobs.squeeze(1)
            copy_logprobs = copy_logprobs.squeeze(1)

            generation_logprobs = generation_logprobs.reshape(batch_size, beam_size, generation_logprobs.shape[-1])
            copy_logprobs = copy_logprobs.reshape(batch_size, beam_size, copy_logprobs.shape[-1])

            prob_scores = torch.zeros([batch_size, beam_size,
                generation_logprobs.shape[-1] + copy_logprobs.shape[-1]], dtype=torch.float32, device=device)
            prob_scores[:, :, :generation_logprobs.shape[-1]] = torch.exp(generation_logprobs)

            # Factoring in the copy scores
            expanded_token_ids = batch_data.input_ids.unsqueeze(1).expand(-1, beam_size, -1)
            prob_scores += scatter_add(src=torch.exp(copy_logprobs), index=expanded_token_ids, out=torch.zeros_like(prob_scores))

            top_scores_per_beam, top_indices_per_beam = torch.topk(prob_scores, k=BEAM_SIZE, dim=-1)
            
            updated_scores = torch.einsum('eb,ebm->ebm', beam_scores, top_scores_per_beam)
            retained_scores = beam_scores.unsqueeze(-1).expand(-1, -1, top_scores_per_beam.shape[-1])

            # Trying to keep at most one ray corresponding to completed beams
            end_mask = (torch.arange(beam_size) == 0).type(torch.float32).to(device)
            end_scores = torch.einsum('b,ebm->ebm', end_mask, retained_scores)
            
            possible_next_scores = torch.where(beam_status.unsqueeze(-1) == 1, end_scores, updated_scores)
            possible_next_status = torch.where(top_indices_per_beam == self.embedding_store.get_end_id(),
                torch.ones([batch_size, beam_size, top_scores_per_beam.shape[-1]], dtype=torch.uint8, device=device),
                beam_status.unsqueeze(-1).expand(-1,-1,top_scores_per_beam.shape[-1]))
            
            possible_beam_predicted_ids = beam_predicted_ids.unsqueeze(2).expand(-1, -1, top_scores_per_beam.shape[-1], -1)
            pool_next_scores = possible_next_scores.reshape(batch_size, -1)
            pool_next_status = possible_next_status.reshape(batch_size, -1)
            pool_next_ids = top_indices_per_beam.reshape(batch_size, -1)
            pool_predicted_ids = possible_beam_predicted_ids.reshape(batch_size, -1, beam_predicted_ids.shape[-1])

            possible_decoder_state = flat_decoder_state.reshape(batch_size, beam_size, flat_decoder_state.shape[-1])
            possible_decoder_state = possible_decoder_state.unsqueeze(2).expand(-1, -1, top_scores_per_beam.shape[-1], -1)
            pool_decoder_state = possible_decoder_state.reshape(batch_size, -1, possible_decoder_state.shape[-1])

            top_scores, top_indices = torch.topk(pool_next_scores, k=BEAM_SIZE, dim=-1)
            next_step_ids = torch.gather(pool_next_ids, -1, top_indices)

            decoder_state = torch.gather(pool_decoder_state, 1, top_indices.unsqueeze(-1).expand(-1,-1, pool_decoder_state.shape[-1]))
            decoder_state = decoder_state.reshape(-1, decoder_state.shape[-1])
            beam_status = torch.gather(pool_next_status, -1, top_indices)
            beam_scores = torch.gather(pool_next_scores, -1, top_indices)

            end_tags = torch.full_like(next_step_ids, self.embedding_store.get_end_id())
            next_step_ids = torch.where(beam_status == 1, end_tags, next_step_ids)

            beam_predicted_ids = torch.gather(pool_predicted_ids, 1, top_indices.unsqueeze(-1).expand(-1, -1, pool_predicted_ids.shape[-1]))
            beam_predicted_ids[:,:,i] = next_step_ids

            unks = torch.full_like(next_step_ids, self.embedding_store.get_nl_id(Vocabulary.get_unk()))
            decoder_input = torch.where(next_step_ids < len(self.embedding_store.nl_vocabulary), next_step_ids, unks).unsqueeze(-1)

        return beam_predicted_ids, beam_scores