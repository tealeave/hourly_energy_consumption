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
        rect, UP, buff=0.2
    )
    shape = Text(shape_text, color=TEXT_COLOR, font_size=20).next_to(
        rect, DOWN, buff=0.2
    )
    return VGroup(rect, label, shape)


# --- Manim Scenes ---

class PatchingConcept(Scene):
    """Visualizes the concept of splitting a time series into patches."""
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
            rect = axes.get_riemann_rectangles(
                graph,
                x_range=[start_x, end_x],
                dx=patch_len,
                color=PATCH_COLOR,
                fill_opacity=0.4,
                stroke_width=1,
                stroke_color=BLACK
            )
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
                patch_rect.get_top(), # Use the actual VGroup element which is the rect VGroup
                embed_box.get_bottom(),
                buff=0.1,
                color=BLACK,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.1
            )
            embedded_patches.add(VGroup(embed_box, embed_label))
            arrow_group.add(arrow)

        embed_title = Text("Patch Embedding (Linear Layer)", color=BLACK, font_size=24)\
            .next_to(embedded_patches, UP, buff=0.3)
        self.play(Write(embed_title), Create(arrow_group), Create(embedded_patches))
        self.wait(2)

        # Conceptually add a global token (like CLS token in ViT)
        global_token_visual = Circle(radius=0.5, color=GLOBAL_TOKEN_COLOR, fill_opacity=0.8)
        global_label = Text("Global\nToken", font_size=18, color=BLACK)\
            .move_to(global_token_visual.get_center())
        global_token_group = VGroup(global_token_visual, global_label)\
            .next_to(embedded_patches, LEFT, buff=0.5) # Position left of embeddings

        self.play(FadeIn(global_token_group, shift=UP))
        # No need to shift again if positioned correctly initially
        self.wait(2)

        # Clear the scene for the next part or end
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


class EmbeddingProcessVisualization(Scene):
    """Visualizes the combination of value, temporal, and positional embeddings."""
    def construct(self):
        self.camera.background_color = WHITE
        title = Text("DataEmbeddingExog: Detailed Process", color=TEXT_COLOR)\
            .scale(0.9).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # --- 1. Input Tensors ---
        input_x = create_tensor_rep("Input Exog (x)", "[B, L, C_exog]", INPUT_COLOR)
        input_x_mark = create_tensor_rep("Input Time (x_mark)", "[B, L, C_time]", INPUT_COLOR)
        inputs_vgroup = VGroup(input_x, input_x_mark).arrange(RIGHT, buff=2.5).shift(UP * 2)

        self.play(FadeIn(inputs_vgroup))
        self.wait(1)

        # --- 2. Value Embedding ---
        arrow_val = Arrow(
            input_x.get_bottom(), input_x.get_bottom() + DOWN * 1.5, buff=0.1, color=TEXT_COLOR
        )
        module_val = Text("TokenEmbedding\n(Linear Layer)", color=TEXT_COLOR, font_size=24)\
            .next_to(arrow_val, RIGHT, buff=0.2)
        output_val = create_tensor_rep("Value Embedding", "[B, L, d_model]", VALUE_EMBED_COLOR)\
            .next_to(arrow_val.get_end(), DOWN, buff=0.5)

        self.play(GrowArrow(arrow_val), Write(module_val))
        self.play(TransformFromCopy(input_x[0], output_val[0]), FadeIn(output_val[1:]))
        self.wait(1)

        # --- 3. Temporal Embedding ---
        # Position output relative to value embedding output
        output_temp = create_tensor_rep("Temporal Embedding", "[B, L, d_model]", TEMPORAL_EMBED_COLOR)
        output_temp.move_to(output_val.get_center() + RIGHT * 3.5)
        # Draw arrow from input to output
        arrow_temp = Arrow(
            input_x_mark.get_bottom(), output_temp.get_top(), buff=0.1, color=TEXT_COLOR
        )
        # Position module label relative to arrow
        module_temp = Text("TimeFeatureEmbedding\n(Linear Layer)", color=TEXT_COLOR, font_size=24)\
            .next_to(arrow_temp, LEFT, buff=0.2)

        self.play(GrowArrow(arrow_temp), Write(module_temp))
        self.play(TransformFromCopy(input_x_mark[0], output_temp[0]), FadeIn(output_temp[1:]))
        self.wait(1)

        # --- 4. Positional Embedding ---
        # Show PE generation concept off to the side
        module_pos = Text("PositionalEmbedding\n(Sin/Cos, Sliced)", color=TEXT_COLOR, font_size=24)\
            .shift(DOWN * 2.5 + LEFT * 5.5) # Adjusted position
        pe_matrix_full = Rectangle(
            width=1.5, height=2.0, color=POS_EMBED_COLOR, fill_opacity=0.2, stroke_width=2
        ).next_to(module_pos, DOWN, buff=0.5)
        pe_matrix_label = Text("PE Matrix\n[max_len, d_model]", font_size=18, color=TEXT_COLOR)\
            .next_to(pe_matrix_full, DOWN, buff=0.1)
        # Represent the slice being taken
        pe_slice = Rectangle(
            width=1.5, height=0.8, color=POS_EMBED_COLOR, fill_opacity=0.6
        ).align_to(pe_matrix_full, UP).shift(DOWN * 0.1)
        pe_slice_label = Text("Slice Used\n[L, d_model]", font_size=18, color=TEXT_COLOR)\
            .next_to(pe_slice, RIGHT, buff=0.2)
        pe_group = VGroup(module_pos, pe_matrix_full, pe_matrix_label, pe_slice, pe_slice_label)

        self.play(FadeIn(pe_group))
        self.wait(1)

        # Represent the final positional embedding tensor used in the sum
        output_pos_rep = create_tensor_rep(
            "Positional Embedding\n(Unsqueezed & Expanded)", "[B, L, d_model]", POS_EMBED_COLOR
        ).move_to(output_val.get_center() + LEFT * 3.5) # Position left of value embed
        output_pos_rep[1].font_size = 18 # Adjust label font size slightly

        self.play(TransformFromCopy(pe_slice, output_pos_rep[0]), FadeIn(output_pos_rep[1:]))
        self.wait(1)

        # --- 5. Combine Embeddings ---
        plus1 = MathTex("+", color=TEXT_COLOR).scale(1.5)\
            .move_to(midpoint(output_pos_rep.get_right(), output_val.get_left()))
        plus2 = MathTex("+", color=TEXT_COLOR).scale(1.5)\
            .move_to(midpoint(output_val.get_right(), output_temp.get_left()))

        embeddings_to_add = VGroup(output_pos_rep, output_val, output_temp)

        # Position the combined result tensor
        output_combined = create_tensor_rep(
            "Combined Embedding", "[B, L, d_model]", COMBINED_EMBED_COLOR, width=3.0
        )
        output_combined.move_to(output_val.get_center() + DOWN * 3.5)
        # Arrow points from the middle of the source embeddings to the result
        arrow_combine = Arrow(
            midpoint(embeddings_to_add.get_bottom(), output_val.get_bottom()), # Midpoint of bases
            output_combined.get_top(),
            buff=0.1, color=TEXT_COLOR
        )

        self.play(Write(plus1), Write(plus2))
        self.play(GrowArrow(arrow_combine))

        # Animate the combination process
        self.play(
            # Indicate inputs fade slightly
            *[emb.animate.set_opacity(0.5) for emb in embeddings_to_add],
            run_time=0.5
        )
        self.play(
            # Transform copies of inputs into the output
             TransformFromCopy(VGroup(*[emb[0] for emb in embeddings_to_add]), output_combined[0]),
             FadeIn(output_combined[1:]),
            # Restore opacity of inputs
             *[emb.animate.set_opacity(1) for emb in embeddings_to_add],
             run_time=1
        )
        self.wait(1)

        # --- 6. Dropout ---
        arrow_dropout = Arrow(
            output_combined.get_bottom(), output_combined.get_bottom() + DOWN * 1.5,
            buff=0.1, color=TEXT_COLOR
        )
        module_dropout = Text("Dropout", color=DROPOUT_COLOR, font_size=24)\
            .next_to(arrow_dropout, RIGHT, buff=0.2)
        final_output = create_tensor_rep(
            "Final Output Embedding", "[B, L, d_model]", COMBINED_EMBED_COLOR
        ).next_to(arrow_dropout.get_end(), DOWN, buff=0.5)

        self.play(GrowArrow(arrow_dropout), Write(module_dropout))
        self.play(TransformFromCopy(output_combined[0], final_output[0]), FadeIn(final_output[1:]))
        self.wait(2)

        # --- Final Polish: Ensure visibility ---
        # Group all animated mobjects
        final_group = VGroup(
            title, inputs_vgroup,
            arrow_val, module_val, output_val,
            arrow_temp, module_temp, output_temp,
            pe_group, output_pos_rep,
            plus1, plus2, arrow_combine, output_combined,
            arrow_dropout, module_dropout, final_output
        )
        # If the final output is too low, scale and center the whole animation
        if final_output.get_bottom()[1] < -config.frame_height / 2 + 0.5:
            self.play(final_group.animate.scale(0.85).move_to(ORIGIN))

        self.wait(3)
        # Optional: Fade out everything at the end
        # self.play(*[FadeOut(mob) for mob in self.mobjects])