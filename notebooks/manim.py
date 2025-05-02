from manim import *
import numpy as np
import pandas as pd # Assuming access to data for visualization

# --- Configuration & Style ---
config.background_color = WHITE
config.frame_width = 16
config.frame_height = 9
# Use a color scheme consistent with the notebook plots if possible
# Example: Blue for actual, Orange for predicted, Green/Red/Purple for splits
ACTUAL_COLOR = BLUE
PREDICTED_COLOR = ORANGE
TRAIN_COLOR = GREEN
VAL_COLOR = YELLOW_E # Using yellow for visibility on white
TEST_COLOR = RED
PATCH_COLOR = PURPLE
EMBED_COLOR = TEAL

# --- Helper Functions (Optional: Load sample data) ---
# In a real scenario, load a small sample of pjme_weather data
# For this example, we'll use dummy data
def load_dummy_data(n_points=500):
    index = pd.date_range(start='2018-01-01', periods=n_points, freq='h') # Changed 'H' to 'h'
    data = np.sin(np.arange(n_points) * 0.1) * 10000 + 30000 + np.random.randn(n_points) * 1000
    return pd.Series(data, index=index, name='PJME_MW')

# --- Manim Scenes ---

class TimeSeriesIntro(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        title = Text("PJM Hourly Load Forecasting", color=BLACK).to_edge(UP)
        subtitle = Text("Visualizing the TimeXer Workflow", color=BLACK).next_to(title, DOWN, buff=0.2)
        self.play(Write(title), Write(subtitle))
        self.wait(1)

        # Load dummy data
        ts_data = load_dummy_data(n_points=168*2) # ~2 weeks

        # Create Axes
        axes = Axes(
            x_range=[0, len(ts_data), 24*7], # Show time steps, label weeks
            y_range=[ts_data.min() * 0.9, ts_data.max() * 1.1, 5000],
            x_length=10,
            y_length=5,
            axis_config={"color": BLACK, "include_numbers": True},
            x_axis_config={"color": BLACK, "label_direction": DOWN},
            y_axis_config={"color": BLACK, "label_direction": LEFT},
            tips=False,
        ).to_edge(DOWN).shift(UP*0.5)
        axes.x_axis.label_converter = lambda x: f"W{int(x / (24*7))}" # Label weeks
        axes.y_axis.label_converter = lambda y: f"{int(y/1000)}k" # Label in kW

        # Create the plot
        graph = axes.plot(lambda x: ts_data.iloc[int(x)], x_range=[0, len(ts_data)-1], use_smoothing=False, color=ACTUAL_COLOR)
        graph_label = axes.get_graph_label(graph, label='PJME Load (MW)', x_val=len(ts_data)*0.8, direction=UP, color=BLACK)

        self.play(Create(axes), Create(graph), Write(graph_label))
        self.wait(2)

        # Store axes and graph for potential use in next scene (if combined)
        self.axes = axes
        self.graph = graph
        self.ts_data = ts_data

        # Transition out or keep for next scene
        # self.play(FadeOut(title), FadeOut(subtitle), FadeOut(axes), FadeOut(graph), FadeOut(graph_label))


class DataSplitting(TimeSeriesIntro): # Inherit to reuse the plot
    def construct(self):
        super().construct() # Run the intro animation first

        axes = self.axes
        graph = self.graph
        ts_data = self.ts_data
        n_points = len(ts_data)

        # Define split points (adjust based on dummy data length)
        val_split_idx = int(n_points * 0.8)
        test_split_idx = int(n_points * 0.9)

        # Create rectangles to highlight splits
        train_rect = axes.get_riemann_rectangles(
            graph, x_range=[0, val_split_idx], dx=(val_split_idx), color=TRAIN_COLOR, fill_opacity=0.3, stroke_width=0
        )
        val_rect = axes.get_riemann_rectangles(
            graph, x_range=[val_split_idx, test_split_idx], dx=(test_split_idx - val_split_idx), color=VAL_COLOR, fill_opacity=0.4, stroke_width=0
        )
        test_rect = axes.get_riemann_rectangles(
            graph, x_range=[test_split_idx, n_points], dx=(n_points - test_split_idx), color=TEST_COLOR, fill_opacity=0.5, stroke_width=0
        )

        # Labels for splits
        train_label = Text("Train (80%)", font_size=24, color=TRAIN_COLOR).next_to(axes.c2p(val_split_idx * 0.4, ts_data.max()), UP)
        val_label = Text("Val (10%)", font_size=24, color=VAL_COLOR).next_to(axes.c2p(val_split_idx + (test_split_idx - val_split_idx) * 0.5, ts_data.max()), UP)
        test_label = Text("Test (10%)", font_size=24, color=TEST_COLOR).next_to(axes.c2p(test_split_idx + (n_points - test_split_idx) * 0.5, ts_data.max()), UP)

        self.play(
            FadeIn(train_rect), Write(train_label),
            FadeIn(val_rect), Write(val_label),
            FadeIn(test_rect), Write(test_label)
        )
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects]) # Clear scene


class PatchingConcept(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        title = Text("TimeXer: Patching", color=BLACK).to_edge(UP)
        self.play(Write(title))

        # Simplified time series segment
        seq_len = 48 # Example sequence length
        patch_len = 12 # Example patch length
        num_patches = seq_len // patch_len
        y_vals = np.sin(np.arange(seq_len) * 0.2) * 2 + np.random.randn(seq_len)*0.2

        axes = Axes(
            x_range=[0, seq_len, patch_len], y_range=[-3, 3, 1],
            x_length=10, y_length=4,
            axis_config={"color": BLACK, "include_numbers": True},
            tips=False,
        ).shift(DOWN*1)
        graph = axes.plot(lambda x: y_vals[int(x)], x_range=[0, seq_len-1], use_smoothing=False, color=ACTUAL_COLOR)
        graph_label = Text("Input Sequence (seq_len)", color=BLACK, font_size=24).next_to(axes, UP)

        self.play(Create(axes), Create(graph), Write(graph_label))
        self.wait(1)

        # Show patches
        patches = VGroup()
        patch_labels = VGroup()
        for i in range(num_patches):
            start_x = i * patch_len
            end_x = (i + 1) * patch_len
            rect = axes.get_riemann_rectangles(
                graph, x_range=[start_x, end_x], dx=patch_len, color=PATCH_COLOR, fill_opacity=0.4, stroke_width=1, stroke_color=BLACK
            )
            label = Text(f"P{i+1}", font_size=20, color=BLACK).move_to(axes.c2p(start_x + patch_len/2, -2.5))
            patches.add(rect)
            patch_labels.add(label)

        self.play(Create(patches), Write(patch_labels))
        self.wait(1)

        # Show embedding concept
        embedded_patches = VGroup()
        arrow_group = VGroup()
        for i, patch_rect in enumerate(patches):
            embed_box = Rectangle(width=0.8, height=1.5, color=EMBED_COLOR, fill_opacity=0.7).shift(UP*2 + LEFT*4 + RIGHT*i*1.2)
            embed_label = Text(f"Emb{i+1}", font_size=18, color=WHITE).move_to(embed_box.get_center())
            arrow = Arrow(patch_rect.get_top(), embed_box.get_bottom(), buff=0.1, color=BLACK, stroke_width=2, max_tip_length_to_length_ratio=0.1)
            embedded_patches.add(VGroup(embed_box, embed_label))
            arrow_group.add(arrow)

        embed_title = Text("Patch Embedding (Linear Layer)", color=BLACK, font_size=24).next_to(embedded_patches, UP)
        self.play(Write(embed_title), Create(arrow_group), Create(embedded_patches))
        self.wait(2)

        # Add Global Token Concept
        global_token = Circle(radius=0.5, color=GOLD, fill_opacity=0.8).move_to(embedded_patches.get_center() + LEFT * 1.5 + DOWN*0.2)
        global_label = Text("Global\nToken", font_size=18, color=BLACK).move_to(global_token.get_center())
        global_vg = VGroup(global_token, global_label)

        self.play(FadeIn(global_vg, shift=UP))
        self.play(global_vg.animate.shift(LEFT*3)) # Move it to the start conceptually
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects]) # Clear scene


class ForecastVisualization(Scene):
     def construct(self):
        self.camera.background_color = WHITE
        title = Text("Forecast vs. Actuals", color=BLACK).to_edge(UP)
        self.play(Write(title))

        # Dummy forecast data (e.g., first 168 hours of test)
        n_points = 168
        actual_data = load_dummy_data(n_points)
        # Simulate predictions (e.g., actuals + noise, maybe slightly biased)
        predicted_data = actual_data + np.random.randn(n_points) * 1500 - 500 # Add noise and slight under-prediction

        axes = Axes(
            x_range=[0, n_points, 24], # Show hours, label days
            y_range=[min(actual_data.min(), predicted_data.min()) * 0.9, max(actual_data.max(), predicted_data.max()) * 1.1, 5000],
            x_length=12,
            y_length=6,
            axis_config={"color": BLACK, "include_numbers": True},
            x_axis_config={"color": BLACK, "label_direction": DOWN},
            y_axis_config={"color": BLACK, "label_direction": LEFT},
            tips=False,
        ).shift(DOWN*0.5)
        axes.x_axis.label_converter = lambda x: f"Day {int(x / 24)}" # Label days
        axes.y_axis.label_converter = lambda y: f"{int(y/1000)}k" # Label in kW

        actual_graph = axes.plot(lambda x: actual_data.iloc[int(x)], x_range=[0, n_points-1], use_smoothing=False, color=ACTUAL_COLOR)
        predicted_graph = axes.plot(lambda x: predicted_data.iloc[int(x)], x_range=[0, n_points-1], use_smoothing=False, color=PREDICTED_COLOR, stroke_width=2) # Removed stroke_dashlength

        actual_label = Text("Actual", color=ACTUAL_COLOR, font_size=24).next_to(axes.c2p(n_points*0.8, actual_data.max()), UP, buff=0.1)
        predicted_label = Text("Predicted", color=PREDICTED_COLOR, font_size=24).next_to(actual_label, DOWN, buff=0.2)

        self.play(Create(axes))
        self.play(Create(actual_graph), Write(actual_label), run_time=2)
        self.play(Create(predicted_graph), Write(predicted_label), run_time=2)
        self.wait(3)

        # Optional: Show metrics (RMSE, MAE, MAPE)
        # rmse_val = np.sqrt(np.mean((actual_data - predicted_data)**2))
        # mae_val = np.mean(np.abs(actual_data - predicted_data))
        # mape_val = np.mean(np.abs((actual_data - predicted_data) / actual_data)) * 100
        # metrics_text = VGroup(
        #     Text(f"RMSE: {rmse_val:.0f} MW", color=BLACK, font_size=20),
        #     Text(f"MAE:  {mae_val:.0f} MW", color=BLACK, font_size=20),
        #     Text(f"MAPE: {mape_val:.2f}%", color=BLACK, font_size=20),
        # ).arrange(DOWN, aligned_edge=LEFT).to_corner(UR).shift(LEFT*0.5)
        # self.play(Write(metrics_text))
        # self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects]) # Clear scene

# To render a specific scene, you would run from the terminal:
# manim -pql manim.py TimeSeriesIntro
# manim -pql manim.py DataSplitting
# manim -pql manim.py PatchingConcept
# manim -pql manim.py ForecastVisualization
# Or combine them into one scene if desired.

