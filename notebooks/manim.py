# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from manim import *

# --- Configuration & Style ---
config.background_color = WHITE
config.frame_width = 18  # Use the wider setting for EmbeddingProcessVisualization
config.frame_height = 10
TEXT_COLOR = BLACK

# Color definitions (consolidated)
ACTUAL_COLOR = BLUE
PREDICTED_COLOR = ORANGE
TRAIN_COLOR = GREEN
VAL_COLOR = YELLOW_E
TEST_COLOR = RED
PATCH_COLOR = PURPLE
EMBED_COLOR = TEAL  # Used in PatchingConcept
VALUE_EMBED_COLOR = BLUE_E
TEMPORAL_EMBED_COLOR = GREEN_E
POS_EMBED_COLOR = PURPLE_E
COMBINED_EMBED_COLOR = TEAL_E # Used in EmbeddingProcessVisualization
INPUT_COLOR = GREY_BROWN
DROPOUT_COLOR = RED_E
GLOBAL_TOKEN_COLOR = GOLD

# --- NEW Colors for Overall Architecture ---
ENDOG_INPUT_COLOR = BLUE_C
EXOG_INPUT_COLOR = GREEN_C
ENDOG_EMBED_COLOR = BLUE_D
EXOG_EMBED_COLOR = GREEN_D
ENCODER_COLOR = GREY_D # Slightly darker grey for the main block
HEAD_COLOR = ORANGE # <<< FIXED LINE: Changed ORANGE_E to ORANGE
FORECAST_COLOR = MAROON_B
ATTENTION_SELF_COLOR = PINK
ATTENTION_CROSS_COLOR = LIGHT_BROWN


# --- Helper Functions ---

def load_dummy_data(n_points=500):
    """Generates simple dummy time series data."""
    index = pd.date_range(start='2018-01-01', periods=n_points, freq='h')
    data = np.sin(np.arange(n_points) * 0.1) * 10000 + 30000 + \
           np.random.randn(n_points) * 1000
    return pd.Series(data, index=index, name='PJME_MW')

def create_tensor_rep(label_text, shape_text, color, width=2.5, height=1.5):
    """Creates a VGroup representing a tensor with label and shape."""
    rect = Rectangle(
        width=width, height=height, color=color, fill_color=color, fill_opacity=0.6
    )
    label = Text(label_text, color=TEXT_COLOR, font_size=24).next_to(
        rect, UP, buff=0.25 # Slightly increased buffer
    )
    shape = Text(shape_text, color=TEXT_COLOR, font_size=20).next_to(
        rect, DOWN, buff=0.25 # Slightly increased buffer
    )
    return VGroup(rect, label, shape)


# --- Manim Scenes ---

class PatchingConcept(Scene):
    """Visualizes the concept of splitting a time series into patches."""
    # (Keep the PatchingConcept class exactly as it was in the previous version)
    def construct(self):
        self.camera.background_color = WHITE
        title = Text("Time Series: Patching Concept", color=BLACK).to_edge(UP)
        self.play(Write(title))

        # Parameters for the visualization
        seq_len = 48
        patch_len = 12
        num_patches = seq_len // patch_len
        y_vals = np.sin(np.arange(seq_len) * 0.2) * 2 + \
                 np.random.randn(seq_len) * 0.2

        # Create axes and plot the time series graph
        axes = Axes(
            x_range=[0, seq_len, patch_len],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=4,
            axis_config={"color": BLACK, "include_numbers": True},
            tips=False,
        ).shift(DOWN * 1)
        # Ensure graph plots points accurately
        graph = axes.plot(
            lambda x: y_vals[int(x)] if 0 <= x < seq_len else 0,
            x_range=[0, seq_len - 1e-9], # Use small epsilon for range end
            use_smoothing=False,
            color=ACTUAL_COLOR
        )
        graph_label = Text("Input Sequence (seq_len)", color=BLACK, font_size=24)\
            .next_to(axes, UP)

        self.play(Create(axes), Create(graph), Write(graph_label))
        self.wait(1)

        # Create visual representations of patches
        patches_visual = VGroup()
        patch_labels = VGroup()
        for i in range(num_patches):
            start_x = i * patch_len
            end_x = (i + 1) * patch_len
            # Use list comprehension to handle potential empty rectangles if graph is flat
            rect_list = axes.get_riemann_rectangles(
                graph,
                x_range=[start_x, end_x],
                dx=patch_len,
                color=PATCH_COLOR,
                fill_opacity=0.4,
                stroke_width=1,
                stroke_color=BLACK
            )
            if not rect_list: continue # Skip if no rectangle generated
            rect = rect_list[0] # Assume first rectangle is representative if multiple generated per patch

            # Position label below the patch rectangle
            label = Text(f"P{i+1}", font_size=20, color=BLACK).move_to(
                axes.c2p(start_x + patch_len / 2, axes.y_range[0] - 0.5) # Below x-axis
            )
            patches_visual.add(rect)
            patch_labels.add(label)

        self.play(Create(patches_visual), Write(patch_labels))
        self.wait(1)

        # Show the concept of embedding each patch
        embedded_patches = VGroup()
        arrow_group = VGroup()
        embed_base_pos = UP * 2.5 + LEFT * 4 # Starting position for embedded boxes

        for i, patch_rect in enumerate(patches_visual):
            embed_box = Rectangle(
                width=0.8, height=1.5, color=EMBED_COLOR, fill_opacity=0.7
            ).move_to(embed_base_pos + RIGHT * i * 1.2)
            embed_label = Text(f"Emb{i+1}", font_size=18, color=WHITE)\
                .move_to(embed_box.get_center())
            # Ensure arrows originate from the top center of the patch rectangles
            arrow = Arrow(
                patch_rect.get_top(),
                embed_box.get_bottom(),
                buff=0.1,
                color=BLACK,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.1
            )
            embedded_patches.add(VGroup(embed_box, embed_label))
            arrow_group.add(arrow)

        embed_title = Text("Patch Embedding (Linear Layer)", color=BLACK, font_size=24)\
            .next_to(embedded_patches, UP, buff=0.5)
        self.play(Write(embed_title), Create(arrow_group), Create(embedded_patches))
        self.wait(2)

        # Conceptually add a global token (like CLS token in ViT)
        global_token_visual = Circle(radius=0.5, color=GLOBAL_TOKEN_COLOR, fill_opacity=0.8)
        global_label = Text("Global\nToken", font_size=18, color=BLACK)\
            .move_to(global_token_visual.get_center())
        global_token_group = VGroup(global_token_visual, global_label)\
            .next_to(embedded_patches, LEFT, buff=0.5) # Position left of embeddings

        self.play(FadeIn(global_token_group, shift=UP))
        self.wait(1) # Shortened wait

        # --- Start: Added section for global token interaction ---

        # 1. Show sequence formation (Global Token + Patch Embeddings)
        input_sequence_group = VGroup(global_token_group, embedded_patches).copy()
        input_sequence_label = Text("Input Sequence to Transformer", color=BLACK, font_size=20)\
            .next_to(input_sequence_group, UP)

        self.play(
            input_sequence_group.animate.arrange(RIGHT, buff=0.2).next_to(embed_title, DOWN, buff=1.0),
            Write(input_sequence_label)
        )
        self.wait(1.5)

        # 2. Abstract Transformer Encoder block
        encoder_box = Rectangle(
            width=input_sequence_group.width + 1, height=2.0, color=GREY_BROWN, fill_opacity=0.2
        ).move_to(input_sequence_group.get_center() + DOWN * 2.5)
        encoder_label = Text("Transformer Encoder", color=BLACK, font_size=24)\
            .move_to(encoder_box.get_center())
        encoder_group = VGroup(encoder_box, encoder_label)

        arrow_to_encoder = Arrow(
            input_sequence_group.get_bottom(), encoder_box.get_top(), buff=0.1, color=BLACK
        )

        self.play(FadeIn(encoder_group), GrowArrow(arrow_to_encoder))
        self.wait(1)

        # 3. Show information flow (abstractly) - patches influence global token
        # Copy the global token and patch embeddings inside the encoder concept
        processed_sequence = input_sequence_group.copy().move_to(encoder_box.get_center())
        processed_global = processed_sequence[0]
        processed_patches = processed_sequence[1]

        # Arrows from patches to global token
        attention_arrows = VGroup(*[
            Arrow(
                patch.get_center(),
                processed_global.get_center(),
                buff=0.1,
                color=TEAL, # Use embedding color for attention flow
                stroke_width=1.5,
                max_tip_length_to_length_ratio=0.1
            ) for patch in processed_patches
        ])
        attention_text = Text("Global token aggregates info\nvia self-attention", color=TEAL, font_size=18)\
            .next_to(encoder_box, RIGHT, buff=0.3)

        self.play(FadeIn(processed_sequence, attention_arrows, attention_text))
        # Indicate processing/aggregation with a pulse on the global token
        self.play(Indicate(processed_global, color=GLOBAL_TOKEN_COLOR, scale_factor=1.2))
        self.wait(2)
        self.play(FadeOut(attention_arrows, attention_text)) # Clean up attention visuals

        # 4. Output: Final Global Token Embedding is used
        final_global_token = processed_global.copy()
        arrow_from_encoder = Arrow(
            encoder_box.get_bottom(), encoder_box.get_bottom() + DOWN * 1.0, buff=0.1, color=BLACK
        )
        final_global_token.next_to(arrow_from_encoder.get_end(), DOWN, buff=0.1)
        final_global_label = Text("Final Global Embedding\n(Used for Prediction)", color=BLACK, font_size=18)\
            .next_to(final_global_token, DOWN, buff=0.2)

        self.play(
            GrowArrow(arrow_from_encoder),
            TransformFromCopy(processed_global, final_global_token),
            FadeOut(processed_patches), # Fade out patch embeddings as focus shifts
            Write(final_global_label)
        )
        self.wait(2)

        # 5. Prediction Head (Optional visualization)
        pred_head_box = Rectangle(width=2, height=1, color=BLUE, fill_opacity=0.3)\
            .next_to(final_global_token, RIGHT, buff=1.0)
        pred_head_label = Text("Prediction\nHead", color=BLACK, font_size=18)\
            .move_to(pred_head_box.get_center())
        pred_head_group = VGroup(pred_head_box, pred_head_label)
        arrow_to_pred = Arrow(
            final_global_token.get_right(), pred_head_box.get_left(), buff=0.1, color=BLACK
        )

        self.play(FadeIn(pred_head_group), GrowArrow(arrow_to_pred))
        self.wait(2)

        # --- End: Added section ---

        # Clear the scene (adjust which objects to fade out)
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob is not title]) # Keep title
        self.wait(1)


# Inherit from MovingCameraScene to allow camera movement
class EmbeddingProcessVisualization(MovingCameraScene):
    """Visualizes the combination of value, temporal, and positional embeddings."""
    def construct(self):
        self.camera.background_color = WHITE
        # Save initial camera state if needed later
        self.camera.frame.save_state()

        title = Text("DataEmbeddingExog: Detailed Process", color=TEXT_COLOR)\
            .scale(0.9).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # --- 1. Input Tensors ---
        input_x = create_tensor_rep("Input Exog (x)", "[B, L, C_exog]", INPUT_COLOR)
        input_x_mark = create_tensor_rep("Input Time (x_mark)", "[B, L, C_time]", INPUT_COLOR)
        inputs_vgroup = VGroup(input_x, input_x_mark).arrange(RIGHT, buff=3.0).shift(UP * 2.5) # Increased buff, shifted slightly higher

        self.play(FadeIn(inputs_vgroup))
        self.wait(1)

        # --- 2. Value Embedding ---
        arrow_val = Arrow(
            input_x.get_bottom(), input_x.get_bottom() + DOWN * 1.5, buff=0.1, color=TEXT_COLOR
        )
        # Adjust module label position for less overlap
        module_val = Text("TokenEmbedding\n(Linear Layer)", color=TEXT_COLOR, font_size=24)\
            .next_to(arrow_val, RIGHT, buff=0.4) # Increased buffer
        output_val = create_tensor_rep("Value Embedding", "[B, L, d_model]", VALUE_EMBED_COLOR)\
            .next_to(arrow_val.get_end(), DOWN, buff=0.75) # Increased buffer

        self.play(GrowArrow(arrow_val), Write(module_val))
        self.play(TransformFromCopy(input_x[0], output_val[0]), FadeIn(output_val[1:]))
        self.wait(1)

        # --- 3. Temporal Embedding ---
        # Position output relative to value embedding output
        output_temp = create_tensor_rep("Temporal Embedding", "[B, L, d_model]", TEMPORAL_EMBED_COLOR)
        output_temp.move_to(output_val.get_center() + RIGHT * 4.0) # Increased spacing
        # Draw arrow from input to output
        arrow_temp = Arrow(
            input_x_mark.get_bottom(), output_temp.get_top(), buff=0.1, color=TEXT_COLOR
        )
        # Adjust module label position
        module_temp = Text("TimeFeatureEmbedding\n(Linear Layer)", color=TEXT_COLOR, font_size=24)\
            .next_to(arrow_temp, RIGHT) # Increased buffer

        self.play(GrowArrow(arrow_temp), Write(module_temp))
        self.play(TransformFromCopy(input_x_mark[0], output_temp[0]), FadeIn(output_temp[1:]))
        self.wait(1)

        # --- 4. Positional Embedding ---
        # Keep PE generation concept off to the far side
        module_pos = Text("PositionalEmbedding\n(Sin/Cos, Sliced)", color=TEXT_COLOR, font_size=22)\
            .shift(DOWN * 2.0 + LEFT * 6.0) # Adjusted position, smaller font
        pe_matrix_full = Rectangle(
            width=1.5, height=2.0, color=POS_EMBED_COLOR, fill_opacity=0.2, stroke_width=2
        ).next_to(module_pos, DOWN, buff=0.4)
        pe_matrix_label = Text("PE Matrix\n[max_len, d_model]", font_size=16, color=TEXT_COLOR)\
            .next_to(pe_matrix_full, DOWN, buff=0.2)
        pe_slice = Rectangle(
            width=1.5, height=0.8, color=POS_EMBED_COLOR, fill_opacity=0.6
        ).align_to(pe_matrix_full, UP).shift(DOWN * 0.1)
        pe_slice_label = Text("Slice Used\n[L, d_model]", font_size=16, color=TEXT_COLOR)\
            .next_to(pe_slice, DOWN, buff=0.3) # Increased buffer
        pe_group = VGroup(module_pos, pe_matrix_full, pe_matrix_label, pe_slice, pe_slice_label)

        self.play(FadeIn(pe_group))
        self.wait(1)

        # Represent the final positional embedding tensor used in the sum
        output_pos_rep = create_tensor_rep(
            "Positional Embedding\n(Unsqueezed & Expanded)", "[B, L, d_model]", POS_EMBED_COLOR
        ).move_to(output_val.get_center() + LEFT * 4.0) # Increased spacing
        output_pos_rep[1].font_size = 18 # Adjust label font size slightly

        self.play(TransformFromCopy(pe_slice, output_pos_rep[0]), FadeIn(output_pos_rep[1:]))
        self.wait(1)

        # --- Camera Pan 1 ---
        # Move camera to focus on the area where embeddings will be combined
        embeddings_to_add = VGroup(output_pos_rep, output_val, output_temp)
        self.play(
            self.camera.frame.animate.move_to(embeddings_to_add.get_center() + DOWN * 1.0),
            run_time=1.5 # Smooth pan
        )
        self.wait(0.5)


        # --- 5. Combine Embeddings ---
        plus1 = MathTex("+", color=TEXT_COLOR).scale(1.5)\
            .move_to(midpoint(output_pos_rep.get_right(), output_val.get_left()))
        plus2 = MathTex("+", color=TEXT_COLOR).scale(1.5)\
            .move_to(midpoint(output_val.get_right(), output_temp.get_left()))

        # Position the combined result tensor relative to current view
        output_combined = create_tensor_rep(
            "Combined Embedding", "[B, L, d_model]", COMBINED_EMBED_COLOR, width=3.0
        )
        # Position below the center of the current group
        output_combined.move_to(embeddings_to_add.get_center() + DOWN * 3.0)

        arrow_combine = Arrow(
            embeddings_to_add.get_bottom(), # Arrow starts from bottom center of the group
            output_combined.get_top(),
            buff=0.2, color=TEXT_COLOR # Increased buffer
        )

        self.play(Write(plus1), Write(plus2))
        self.play(GrowArrow(arrow_combine))

        # Animate the combination process
        self.play(
            *[emb.animate.set_opacity(0.5) for emb in embeddings_to_add],
            run_time=0.5
        )
        self.play(
              TransformFromCopy(VGroup(*[emb[0] for emb in embeddings_to_add]), output_combined[0]),
              FadeIn(output_combined[1:]),
              *[emb.animate.set_opacity(1) for emb in embeddings_to_add],
              run_time=1
        )
        self.wait(1)

        # --- Camera Pan 2 ---
        # Move camera to focus on the combined output before dropout
        self.play(
            self.camera.frame.animate.move_to(output_combined.get_center() + DOWN * 1.0),
              run_time=1.5 # Smooth pan
        )
        self.wait(0.5)

        # --- 6. Dropout ---
        arrow_dropout = Arrow(
            output_combined.get_bottom(), output_combined.get_bottom() + DOWN * 1.5,
            buff=0.1, color=TEXT_COLOR
        )
        # Adjust module label position
        module_dropout = Text("Dropout", color=DROPOUT_COLOR, font_size=24)\
            .next_to(arrow_dropout, RIGHT, buff=0.4) # Increased buffer
        final_output = create_tensor_rep(
            "Final Output Embedding", "[B, L, d_model]", COMBINED_EMBED_COLOR
        ).next_to(arrow_dropout.get_end(), DOWN, buff=0.2) # Increased buffer

        self.play(GrowArrow(arrow_dropout), Write(module_dropout))
        self.play(TransformFromCopy(output_combined[0], final_output[0]), FadeIn(final_output[1:]))
        self.wait(1)

        # --- Final View ---
        # Ensure the final output is nicely framed
        self.play(
            self.camera.frame.animate.move_to(final_output.get_center()),
            run_time=1.0
            )

        # Removed the final scaling logic

        self.wait(3)
        # Optional: Restore camera or fade out
        # self.play(Restore(self.camera.frame))
        # self.play(*[FadeOut(mob) for mob in self.mobjects])

# --- NEW SCENE START ---

# --- NEW SCENE START ---

class TimeXerOverallArchitecture(Scene):
    """Visualizes the high-level architecture of the TimeXer model."""
    def construct(self):
        self.camera.background_color = WHITE
        title = Text("TimeXer: Overall Architecture", color=TEXT_COLOR).scale(0.9).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # --- 1. Inputs ---
        # Endogenous Input (Target Variable Series, e.g., PJME_MW)
        # Assumes univariate target, shape [B, L, 1] after dataset processing
        input_endog = create_tensor_rep("Endogenous Input\n(Target History)", "[B, L, 1]", ENDOG_INPUT_COLOR, width=2.8, height=1.8)

        # Exogenous Inputs (Weather, Calendar Features, Time Features)
        # Combining x_enc (exog part) and x_mark_enc conceptually
        input_exog = create_tensor_rep("Exogenous Inputs\n(Weather, Calendar)", "[B, L, C_exog+C_time]", EXOG_INPUT_COLOR, width=3.2, height=1.8)

        inputs_group = VGroup(input_endog, input_exog).arrange(RIGHT, buff=2.5).shift(UP * 2.0)
        self.play(FadeIn(inputs_group))
        self.wait(1)

        # --- 2. Embedding Layers ---
        # Endogenous Embedding Block (Patching + Global Token)
        embed_block_endog = Rectangle(width=3.0, height=1.2, color=ENDOG_EMBED_COLOR, fill_opacity=0.3).next_to(input_endog, DOWN, buff=1.0)
        embed_label_endog = Text("Endog Embedding\n(Patching+Global)", color=TEXT_COLOR, font_size=20).move_to(embed_block_endog.get_center())
        endog_embed_group = VGroup(embed_block_endog, embed_label_endog)
        arrow_to_endog_embed = Arrow(input_endog.get_bottom(), embed_block_endog.get_top(), buff=0.1, color=BLACK)

        # Exogenous Embedding Block (Value + Time + Positional)
        embed_block_exog = Rectangle(width=3.5, height=1.2, color=EXOG_EMBED_COLOR, fill_opacity=0.3).next_to(input_exog, DOWN, buff=1.0)
        embed_label_exog = Text("Exog Embedding\n(Value+Time+Pos)", color=TEXT_COLOR, font_size=20).move_to(embed_block_exog.get_center())
        exog_embed_group = VGroup(embed_block_exog, embed_label_exog)
        arrow_to_exog_embed = Arrow(input_exog.get_bottom(), embed_block_exog.get_top(), buff=0.1, color=BLACK)

        self.play(
            GrowArrow(arrow_to_endog_embed), FadeIn(endog_embed_group),
            GrowArrow(arrow_to_exog_embed), FadeIn(exog_embed_group)
        )
        self.wait(1)

        # --- 3. Embedded Representations (Outputs of Embedding) ---
        # Endogenous Output (Patch Tokens + Global Token) - Shape [B*n_vars_en, N+1, D] -> [B, N+1, D] for n_vars_en=1
        embed_out_endog = create_tensor_rep("Patch & Global Tokens", "[B, N+1, D]", ENDOG_EMBED_COLOR, width=3.0, height=1.5)
        embed_out_endog.next_to(embed_block_endog, DOWN, buff=1.0)
        arrow_from_endog_embed = Arrow(embed_block_endog.get_bottom(), embed_out_endog.get_top(), buff=0.1, color=BLACK)

        # Exogenous Output (Used as Keys/Values in Cross-Attention) - Shape [B, L, D]
        embed_out_exog = create_tensor_rep("Exog Embeddings\n(for Cross-Attn)", "[B, L, D]", EXOG_EMBED_COLOR, width=3.5, height=1.5)
        embed_out_exog.next_to(embed_block_exog, DOWN, buff=1.0)
        arrow_from_exog_embed = Arrow(embed_block_exog.get_bottom(), embed_out_exog.get_top(), buff=0.1, color=BLACK)

        # --- MODIFIED ANIMATION ---
        # First, draw the arrows pointing from the embedding blocks
        self.play(
            GrowArrow(arrow_from_endog_embed),
            GrowArrow(arrow_from_exog_embed)
            )
        # Then, fade in the resulting embedded tensors (replacing TransformFromCopy)
        self.play(
            FadeIn(embed_out_endog),
            FadeIn(embed_out_exog)
            )
        # --- END MODIFIED ANIMATION ---
        self.wait(1.5) # Keep the wait time


        # --- 4. Transformer Encoder Block ---
        encoder_block = Rectangle(width=7.0, height=2.5, color=ENCODER_COLOR, fill_opacity=0.2)
        encoder_block.move_to(midpoint(embed_out_endog.get_bottom(), embed_out_exog.get_bottom()) + DOWN * 2.0)
        encoder_label = Text("TimeXer Encoder Layers", color=TEXT_COLOR, font_size=28).move_to(encoder_block.get_center())
        encoder_group = VGroup(encoder_block, encoder_label)

        # Arrows into the Encoder
        arrow_endog_to_encoder = Arrow(embed_out_endog.get_bottom(), encoder_block.get_top() + LEFT*1.5, buff=0.1, color=BLACK)
        arrow_exog_to_encoder = Arrow(embed_out_exog.get_bottom(), encoder_block.get_top() + RIGHT*1.5, buff=0.1, color=BLACK)

        self.play(
            FadeIn(encoder_group),
            GrowArrow(arrow_endog_to_encoder),
            GrowArrow(arrow_exog_to_encoder)
        )
        self.wait(1)

        # --- 4a. Illustrate Attention inside Encoder (Conceptually) ---
        # Self-Attention within Endogenous Tokens
        self_attn_label = Text("Self-Attention\n(Endog Tokens)", color=ATTENTION_SELF_COLOR, font_size=18)\
            .move_to(encoder_block.get_center() + LEFT * 2.0 + UP * 0.5)
        self_attn_loop = Arc(radius=0.5, start_angle=PI / 2, angle=-1.5 * PI, color=ATTENTION_SELF_COLOR)\
            .move_to(encoder_block.get_center() + LEFT * 2.0 + DOWN * 0.5) # Below label
        self_attn_arrow = Arrow(self_attn_loop.get_start(), self_attn_loop.get_end(),
                                color=ATTENTION_SELF_COLOR, buff=0.05, stroke_width=2, max_tip_length_to_length_ratio=0.2)

        # Cross-Attention from Endogenous Global Token to Exogenous Embeddings
        cross_attn_label = Text("Cross-Attention\n(Global Query, Exog K/V)", color=ATTENTION_CROSS_COLOR, font_size=18)\
            .move_to(encoder_block.get_center() + RIGHT * 1.5 + UP * 0.5)
        # Conceptual start/end points for cross-attention arrow
        cross_start_point = encoder_block.get_center() + LEFT * 0.5 + DOWN * 0.5
        cross_end_point = encoder_block.get_center() + RIGHT * 1.5 + DOWN * 0.5
        cross_attn_arrow = Arrow(cross_start_point, cross_end_point, color=ATTENTION_CROSS_COLOR, buff=0.1, stroke_width=3)

        # FFN Label
        ffn_label = Text("FFN", color=TEXT_COLOR, font_size=18)\
            .move_to(encoder_block.get_center() + DOWN * 0.7)

        self.play(
            Write(self_attn_label), Create(self_attn_loop), Create(self_attn_arrow),
            Write(cross_attn_label), Create(cross_attn_arrow),
            Write(ffn_label)
        )
        self.wait(2.5)
        self.play(
             FadeOut(self_attn_label, self_attn_loop, self_attn_arrow),
             FadeOut(cross_attn_label, cross_attn_arrow),
             FadeOut(ffn_label)
        )
        self.wait(0.5)


        # --- 5. Encoder Output ---
        # Output focuses on the final state of patch/global tokens, especially the global one
        encoder_out_rep = create_tensor_rep("Final Endog Tokens", "[B, N+1, D]", ENDOG_EMBED_COLOR, width=3.0, height=1.5)
        encoder_out_rep.next_to(encoder_block, DOWN, buff=1.5)
        arrow_from_encoder = Arrow(encoder_block.get_bottom(), encoder_out_rep.get_top(), buff=0.1, color=BLACK)

        self.play(GrowArrow(arrow_from_encoder), FadeIn(encoder_out_rep))
        self.wait(1)


        # --- 6. Prediction Head ---
        head_block = Rectangle(width=3.5, height=1.2, color=HEAD_COLOR, fill_opacity=0.3).next_to(encoder_out_rep, DOWN, buff=1.0)
        head_label = Text("Prediction Head\n(Flatten + Linear)", color=TEXT_COLOR, font_size=20).move_to(head_block.get_center())
        head_group = VGroup(head_block, head_label)
        arrow_to_head = Arrow(encoder_out_rep.get_bottom(), head_block.get_top(), buff=0.1, color=BLACK)

        self.play(GrowArrow(arrow_to_head), FadeIn(head_group))
        self.wait(1)

        # --- 7. Final Output (Forecast) ---
        final_forecast = create_tensor_rep("Final Forecast", "[B, pred_len, 1]", FORECAST_COLOR, width=2.8, height=1.5)
        final_forecast.next_to(head_block, DOWN, buff=1.0)
        arrow_to_forecast = Arrow(head_block.get_bottom(), final_forecast.get_top(), buff=0.1, color=BLACK)

        # Also simplify this last step just in case
        self.play(GrowArrow(arrow_to_forecast))
        self.play(FadeIn(final_forecast))
        # Original: self.play(GrowArrow(arrow_to_forecast), TransformFromCopy(head_block, final_forecast))

        self.wait(3)

        # Fade out everything except title maybe
        self.play(*[FadeOut(mob) for mob in self.mobjects if mob is not title])
        self.wait(1)

# --- NEW SCENE END ---