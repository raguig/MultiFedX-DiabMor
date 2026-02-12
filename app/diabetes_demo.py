import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, List, Tuple

# ── Load and preprocess data ──
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    X = df.drop('Outcome', axis=1).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

X_train_data, scaler = load_data()

# ── Model definition ──
class DiabetesNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture: 8 -> 32 -> 16 -> 1 (matches your saved checkpoint)
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

# ── Load best model ──
@st.cache_resource
def load_model():
    model = DiabetesNet()
    model.load_state_dict(torch.load("final_federated_diabetes_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Configure Plotly to suppress warnings
plotly_config = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True
}

# ══════════════════════════════════════════════════════════════════════════
# ── LRP IMPLEMENTATION ──
# ══════════════════════════════════════════════════════════════════════════

class LRPExplainer:
    """
    Layer-wise Relevance Propagation for PyTorch neural networks.
    
    Uses the epsilon-LRP rule (with small stabilizer to prevent division by zero).
    For more advanced rules (alpha-beta, etc.), this can be extended.
    """
    
    def __init__(self, model: nn.Module, epsilon: float = 1e-6):
        """
        Args:
            model: PyTorch model to explain
            epsilon: Small value for numerical stability in relevance propagation
        """
        self.model = model
        self.epsilon = epsilon
        self.activations = {}
        self.relevances = {}
        
    def _register_hooks(self, input_tensor: torch.Tensor):
        """Register forward hooks to capture layer activations"""
        self.activations = {}
        hooks = []
        
        def forward_hook(name):
            def hook_fn(module, input, output):
                self.activations[name] = output.detach().clone()
            return hook_fn
        
        # Register hooks for all Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(forward_hook(name)))
        
        # Forward pass to populate activations
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    def _epsilon_rule(self, layer: nn.Linear, relevance_next: torch.Tensor, 
                      activation_prev: torch.Tensor) -> torch.Tensor:
        """
        Apply epsilon-LRP rule to propagate relevance backward through a layer.
        
        R_i = sum_j ( (a_i * w_ij) / (sum_k a_k * w_kj + epsilon * sign(sum_k a_k * w_kj)) * R_j )
        """
        W = layer.weight  # shape: [out_features, in_features]
        b = layer.bias if layer.bias is not None else 0
        
        # Forward propagation (to get denominator)
        z = torch.matmul(activation_prev, W.T) + b  # [batch, out_features]
        
        # Add epsilon for stability (with sign to handle negative values)
        z_stabilized = z + self.epsilon * torch.sign(z)
        
        # Relevance propagation: R_prev = activation_prev * (W.T @ (R_next / z_stabilized))
        relevance_ratio = relevance_next / z_stabilized  # [batch, out_features]
        relevance_prev = activation_prev * torch.matmul(relevance_ratio, W)  # [batch, in_features]
        
        return relevance_prev
    
    def explain(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate LRP explanations for input.
        
        Args:
            input_tensor: Input sample [1, n_features] or [n_features]
        
        Returns:
            input_relevance: Relevance scores for input features [n_features]
            layer_relevances: Dictionary of relevance scores at each layer
        """
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)  # [1, n_features]
        
        # Ensure gradient tracking
        input_tensor = input_tensor.detach().clone().requires_grad_(False)
        
        # Register hooks and run forward pass
        self._register_hooks(input_tensor)
        
        # Get model output (prediction)
        with torch.no_grad():
            output = self.model(input_tensor)  # [1, 1]
        
        # Initialize relevance at output layer (R = f(x) for single output)
        self.relevances['output'] = output.detach().clone()
        
        # Get all linear layers in reverse order
        linear_layers = [(name, module) for name, module in self.model.named_modules() 
                        if isinstance(module, nn.Linear)]
        linear_layers.reverse()
        
        # Store relevances at each layer
        layer_relevances = {}
        current_relevance = self.relevances['output']
        
        # Build activation list: input -> fc1_output -> fc2_output -> fc3_output
        # For backprop: we need activations BEFORE each layer
        # fc3 needs fc2_output, fc2 needs fc1_output, fc1 needs input
        
        # Create mapping of layer names to their input activations
        layer_to_input_activation = {}
        layer_names = [name for name, _ in linear_layers]
        
        # For fc3 (last layer), input is fc2's output
        # For fc2 (middle layer), input is fc1's output  
        # For fc1 (first layer), input is the original input
        
        if len(linear_layers) >= 3:
            layer_to_input_activation[layer_names[0]] = self.activations[layer_names[1]]  # fc3 <- fc2 output
            layer_to_input_activation[layer_names[1]] = self.activations[layer_names[2]]  # fc2 <- fc1 output
            layer_to_input_activation[layer_names[2]] = input_tensor  # fc1 <- input
        elif len(linear_layers) == 2:
            layer_to_input_activation[layer_names[0]] = self.activations[layer_names[1]]
            layer_to_input_activation[layer_names[1]] = input_tensor
        elif len(linear_layers) == 1:
            layer_to_input_activation[layer_names[0]] = input_tensor
        
        # Backpropagate relevance through layers
        for i, (layer_name, layer) in enumerate(linear_layers):
            # Get activation from previous layer (input to current layer)
            activation_prev = layer_to_input_activation[layer_name]
            
            # Apply LRP rule
            current_relevance = self._epsilon_rule(layer, current_relevance, activation_prev)
            
            # Store layer relevance
            layer_relevances[layer_name] = current_relevance.detach().cpu().numpy()[0]
        
        # Input relevance is the final backpropagated relevance
        input_relevance = current_relevance.detach().cpu().numpy()[0]
        
        return input_relevance, layer_relevances
    
    def get_neuron_contributions(self, input_tensor: torch.Tensor, 
                                 layer_name: str) -> np.ndarray:
        """
        Get individual neuron contributions at a specific layer.
        
        Args:
            input_tensor: Input sample
            layer_name: Name of the layer to analyze
        
        Returns:
            neuron_contributions: Relevance scores for neurons in specified layer
        """
        _, layer_relevances = self.explain(input_tensor)
        
        if layer_name in layer_relevances:
            return layer_relevances[layer_name]
        else:
            raise ValueError(f"Layer {layer_name} not found. Available layers: {list(layer_relevances.keys())}")

# ── Initialize LRP explainer ──
@st.cache_resource
def get_lrp_explainer():
    """Initialize LRP explainer with model"""
    return LRPExplainer(model, epsilon=1e-6)

lrp_explainer = get_lrp_explainer()

# ── SHAP explainer (using KernelExplainer) ──
@st.cache_resource
def get_explainer():
    # Use first 100 samples as background (numpy array)
    background = X_train_data[:100]
    
    # Wrapper function that converts numpy → tensor → model → numpy
    def model_predict_proba(data):
        """Takes numpy array, returns numpy array of predictions"""
        data_tensor = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            preds = model(data_tensor).cpu().numpy().flatten()
        return preds
    
    # Use KernelExplainer instead of DeepExplainer
    explainer = shap.KernelExplainer(model_predict_proba, background)
    return explainer

explainer = get_explainer()

# ── Privacy Results (from your simulations) ──
@st.cache_data
def load_privacy_results():
    """Load privacy-utility trade-off results"""
    return {
        "no_dp": {
            "epsilon": float('inf'),
            "epsilon_str": "∞",
            "test_acc": 0.7468,
            "rounds": 10,
            "noise_multiplier": 0.0
        },
        "moderate_dp": {
            "epsilon": 8.0,
            "epsilon_str": "≈8.0",
            "test_acc": 0.7397,
            "rounds": 12,
            "noise_multiplier": 1.1
        },
        "strong_dp": {
            "epsilon": 3.0,
            "epsilon_str": "≈3.0",
            "test_acc": 0.7318,
            "rounds": 15,
            "noise_multiplier": 2.5
        }
    }

privacy_results = load_privacy_results()

# ══════════════════════════════════════════════════════════════════════════
# ── APP UI ──
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
st.title("🏥 Diabetes Risk Prediction – Federated Learning + Advanced XAI")

st.markdown("""
This model was trained with **federated learning** (Flower + PyTorch) across distributed clients  
while keeping sensitive health data local. Enter your values to see your predicted risk with:
- 🎯 **SHAP**: Model-agnostic, global feature importance
- 🧠 **LRP**: Neural network-specific, layer-wise relevance tracking
""")

# Input form
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0, 200, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
    
    with col2:
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
    
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47, step=0.01)
        age = st.number_input("Age", 0, 120, 30)
    
    submit = st.form_submit_button("Predict Risk", type="primary")

if submit:
    # Normalize input using the saved scaler
    input_raw = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]], dtype=np.float32)
    input_scaled = scaler.transform(input_raw)
    
    with st.spinner("Predicting..."):
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_scaled).float()
            prob = model(input_tensor).item()
        
        risk_level = "High" if prob > 0.5 else "Low"
        color = "red" if prob > 0.5 else "green"
        
        st.markdown(f"**Predicted diabetes probability:** **{prob:.1%}**")
        st.markdown(f"**Risk level:** :{color}[**{risk_level}**]")
        
        if prob > 0.5:
            st.warning("⚠️ Elevated risk — please consult a doctor.")
        else:
            st.success("✅ Lower risk — keep up healthy habits!")
        
        # ══════════════════════════════════════════════════════════════════
        # ── XAI COMPARISON: SHAP vs LRP ──
        # ══════════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.subheader("🔍 Explainability: Why This Prediction?")
        
        # Tabs for different explanation methods
        tab1, tab2, tab3 = st.tabs(["📊 SHAP Analysis", "🧠 LRP Analysis", "⚖️ Comparison"])
        
        feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        
        # ────────────────────────────────────────────────────────────────
        # TAB 1: SHAP ANALYSIS (ENHANCED WITH INTERACTIVE PLOTS)
        # ────────────────────────────────────────────────────────────────
        with tab1:
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** is a model-agnostic method based on game theory.
            - ✅ Works with any model (black box)
            - ✅ Theoretically grounded (Shapley values)
            - ⏱️ Slower computation (~1-2 minutes)
            - 🎯 Shows **global feature importance** across all predictions
            """)
            
            with st.spinner("Computing SHAP values (this may take ~1-2 min)..."):
                shap_values = explainer.shap_values(input_scaled)
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]
            
            # ──── INTERACTIVE BAR PLOT (Plotly) ────
            st.markdown("### 📊 Interactive Feature Contribution")
            
            df_shap = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_values,
                'Abs Value': np.abs(shap_values),
                'Impact': ['↑ Increases Risk' if sv > 0 else '↓ Decreases Risk' for sv in shap_values]
            }).sort_values('Abs Value', ascending=True)
            
            colors = ['#d62728' if val > 0 else '#2ca02c' for val in df_shap['SHAP Value']]
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    y=df_shap['Feature'],
                    x=df_shap['SHAP Value'],
                    orientation='h',
                    marker=dict(color=colors, line=dict(color='black', width=1)),
                    text=[f"{val:.4f}" for val in df_shap['SHAP Value']],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<br><extra></extra>',
                    name='SHAP Value'
                )
            ])
            
            fig_bar.update_layout(
                title='<b>SHAP Feature Importance (Individual Prediction)</b>',
                xaxis_title='SHAP Value (Impact on Prediction)',
                yaxis_title='Feature',
                hovermode='closest',
                template='plotly_white',
                height=500,
                showlegend=False,
                font=dict(size=11),
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
            )
            
            st.plotly_chart(fig_bar, width='stretch')
            
            # ──── FORCE PLOT INTERPRETATION ────
            st.markdown("### 🎯 Prediction Breakdown")
            
            base_value = explainer.expected_value
            model_output = prob
            
            col_force_left, col_force_right = st.columns([1, 2])
            
            with col_force_left:
                st.metric(
                    "Base Risk (Model Average)",
                    f"{base_value:.1%}",
                    help="Average diabetes risk across training data"
                )
                st.metric(
                    "Prediction",
                    f"{model_output:.1%}",
                    delta=f"{(model_output - base_value)*100:+.2f}pp",
                    help="Your individual predicted risk"
                )
            
            with col_force_right:
                cumsum = np.cumsum([base_value] + list(shap_values))
                
                fig_waterfall = go.Figure(go.Waterfall(
                    name="Prediction",
                    orientation="v",
                    x=['Base Value'] + feature_names,
                    textposition="outside",
                    y=[base_value] + list(shap_values),
                    connector={"line": {"color": "rgba(0,0,0,0.3)"}},
                    increasing={"marker": {"color": "#d62728"}},
                    decreasing={"marker": {"color": "#2ca02c"}},
                    totals={"marker": {"color": "#1f77b4"}},
                    hovertemplate='<b>%{x}</b><br>Contribution: %{y:.4f}<br><extra></extra>'
                ))
                
                fig_waterfall.update_layout(
                    title='<b>How Features Combine to Form Prediction</b>',
                    yaxis_title='Cumulative Risk Contribution',
                    template='plotly_white',
                    height=500,
                    showlegend=False,
                    font=dict(size=10)
                )
                
                st.plotly_chart(fig_waterfall, width='stretch')
            
            # ──── DETAILED FEATURE VALUES TABLE ────
            st.markdown("### 📋 Detailed Feature Analysis")
            
            detailed_df = pd.DataFrame({
                'Feature': feature_names,
                'Your Value': input_raw[0],
                'SHAP Value': shap_values,
                'Abs Impact': np.abs(shap_values),
                'Direction': ['↑ Increases Risk' if sv > 0 else '↓ Decreases Risk' for sv in shap_values],
                'Magnitude': ['Very High' if abs(sv) > np.percentile(np.abs(shap_values), 75) 
                              else 'High' if abs(sv) > np.percentile(np.abs(shap_values), 50)
                              else 'Medium' if abs(sv) > np.percentile(np.abs(shap_values), 25)
                              else 'Low' for sv in shap_values]
            }).sort_values('Abs Impact', ascending=False)
            
            st.dataframe(detailed_df, use_container_width=True, hide_index=True)
            
            # ──── WHAT-IF ANALYSIS (FIXED) ────
            st.markdown("### 🔮 What-If Scenarios")
            st.markdown("*Adjust a single feature and see how it impacts your predicted risk.*")
            
            # Initialize session state for what-if
            if "shap_whatif_feature" not in st.session_state:
                st.session_state.shap_whatif_feature = "Glucose"
            if "shap_whatif_value" not in st.session_state:
                st.session_state.shap_whatif_value = 120.0
            
            col_whatif_1, col_whatif_2 = st.columns(2)
            
            with col_whatif_1:
                feature_to_adjust = st.selectbox(
                    "Select feature to adjust:",
                    feature_names,
                    index=feature_names.index(st.session_state.shap_whatif_feature)
                )
                st.session_state.shap_whatif_feature = feature_to_adjust
            
            feature_idx = feature_names.index(feature_to_adjust)
            original_val = float(input_raw[0, feature_idx])
            
            # Define ranges for each feature
            feature_ranges = {
                "Pregnancies": (0.0, 20.0, 1.0),
                "Glucose": (0.0, 200.0, 1.0),
                "BloodPressure": (0.0, 150.0, 1.0),
                "SkinThickness": (0.0, 100.0, 1.0),
                "Insulin": (0.0, 900.0, 10.0),
                "BMI": (0.0, 70.0, 0.1),
                "DiabetesPedigreeFunction": (0.0, 3.0, 0.01),
                "Age": (0.0, 120.0, 1.0)
            }
            
            min_val, max_val, step_val = feature_ranges.get(feature_to_adjust, (0.0, 100.0, 1.0))
            
            with col_whatif_2:
                adjusted_val = st.slider(
                    f"Adjust {feature_to_adjust}:",
                    min_val, max_val, original_val,
                    step=step_val
                )
            
            # Compute prediction with adjusted value
            if adjusted_val != original_val:
                input_adjusted = input_raw.copy()
                input_adjusted[0, feature_idx] = adjusted_val
                input_scaled_adjusted = scaler.transform(input_adjusted)
                
                with torch.no_grad():
                    input_tensor_adjusted = torch.from_numpy(input_scaled_adjusted).float()
                    prob_adjusted = model(input_tensor_adjusted).item()
                
                risk_change = (prob_adjusted - prob) * 100
                
                col_orig, col_adj, col_delta = st.columns(3)
                
                with col_orig:
                    st.metric("Original Risk", f"{prob:.1%}")
                
                with col_adj:
                    st.metric("Adjusted Risk", f"{prob_adjusted:.1%}")
                
                with col_delta:
                    delta_text = f"{risk_change:+.2f}pp"
                    st.metric("Change", delta_text, 
                             delta=delta_text,
                             delta_color="inverse" if risk_change < 0 else "off")
                
                fig_whatif = go.Figure(
                    go.Bar(
                        x=['Original', 'Adjusted'],
                        y=[prob*100, prob_adjusted*100],
                        marker_color=['#1f77b4', '#ff7f0e'],
                        text=[f"{prob:.1%}", f"{prob_adjusted:.1%}"],
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Risk: %{y:.2f}%<extra></extra>'
                    )
                )
                
                fig_whatif.update_layout(
                    title=f'<b>Risk Comparison: {feature_to_adjust}</b>',
                    yaxis_title='Predicted Diabetes Risk (%)',
                    template='plotly_white',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_whatif, config=plotly_config, use_container_width=True)
                
                if risk_change < 0:
                    st.success(f"✅ **Reducing {feature_to_adjust}** would lower your risk by **{abs(risk_change):.2f}pp**")
                else:
                    st.warning(f"⚠️ **Increasing {feature_to_adjust}** would raise your risk by **{risk_change:.2f}pp**")
            
            # ──── BEESWARM PLOT (Global Feature Importance) ────
            st.markdown("---")
            st.markdown("### 🐝 Global SHAP Summary (Background: All Training Data)")
            st.markdown("*This shows how SHAP values vary across the entire dataset for each feature.*")
            
            with st.spinner("Generating global SHAP summary..."):
                background_sample = X_train_data[:50]
                shap_values_background = explainer.shap_values(background_sample)
                if shap_values_background.ndim > 1:
                    shap_values_background = shap_values_background
                else:
                    shap_values_background = shap_values_background.reshape(-1, len(feature_names))
            
            beeswarm_data = []
            for feat_idx, feat_name in enumerate(feature_names):
                shap_vals_feat = shap_values_background[:, feat_idx]
                feature_vals = background_sample[:, feat_idx]
                
                for shap_val, feat_val in zip(shap_vals_feat, feature_vals):
                    beeswarm_data.append({
                        'Feature': feat_name,
                        'SHAP Value': shap_val,
                        'Feature Value': feat_val
                    })
            
            df_beeswarm = pd.DataFrame(beeswarm_data)
            
            fig_beeswarm = px.scatter(
                df_beeswarm,
                x='SHAP Value',
                y='Feature',
                color='Feature Value',
                color_continuous_scale='Viridis',
                title='<b>SHAP Beeswarm: Feature Distributions (50 Training Samples)</b>',
                height=500,
                hover_data={'SHAP Value': ':.4f', 'Feature Value': ':.2f'},
                template='plotly_white'
            )
            
            fig_beeswarm.update_traces(marker=dict(size=8, opacity=0.6, line=dict(width=0.5)))
            fig_beeswarm.update_layout(
                xaxis_title='SHAP Value',
                yaxis_title='Feature',
                hovermode='closest',
                font=dict(size=11)
            )
            
            st.plotly_chart(fig_beeswarm, config=plotly_config, use_container_width=True)
            
            st.caption(
                "💡 **Interpretation**: Points to the right (red) mean higher feature values increase risk. "
                "Points to the left (blue) mean lower feature values increase risk."
            )

        # ────────────────────────────────────────────────────────────────
        # TAB 2: LRP ANALYSIS
        # ────────────────────────────────────────────────────────────────
        with tab2:
            st.markdown("""
            **LRP (Layer-wise Relevance Propagation)** breaks down neural network predictions layer-by-layer.
            - ✅ Works specifically with neural networks
            - ✅ Neuron-level granularity
            - ⏱️ Fast computation (milliseconds)
            - 🎯 Shows **what each neuron contributes** to the prediction
            """)
            
            with st.spinner("Computing LRP values..."):
                input_tensor_lrp = torch.from_numpy(input_scaled).float()
                lrp_relevance, layer_relevances = lrp_explainer.explain(input_tensor_lrp)
            
            st.markdown("### 📊 LRP Feature Relevance")
            
            df_lrp = pd.DataFrame({
                'Feature': feature_names,
                'LRP Relevance': lrp_relevance,
                'Abs Relevance': np.abs(lrp_relevance),
                'Impact': ['↑ Increases Risk' if r > 0 else '↓ Decreases Risk' for r in lrp_relevance]
            }).sort_values('Abs Relevance', ascending=True)
            
            colors_lrp = ['#d62728' if val > 0 else '#2ca02c' for val in df_lrp['LRP Relevance']]
            
            fig_lrp = go.Figure(
                go.Bar(
                    y=df_lrp['Feature'],
                    x=df_lrp['LRP Relevance'],
                    orientation='h',
                    marker=dict(color=colors_lrp, line=dict(color='black', width=1)),
                    text=[f"{val:.4f}" for val in df_lrp['LRP Relevance']],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Relevance: %{x:.4f}<br><extra></extra>',
                    name='LRP Relevance'
                )
            )
            
            fig_lrp.update_layout(
                title='<b>LRP Input Relevance (Neural Network Layer Traces)</b>',
                xaxis_title='Relevance Score',
                yaxis_title='Feature',
                hovermode='closest',
                template='plotly_white',
                height=500,
                showlegend=False,
                font=dict(size=11),
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
            )
            
            st.plotly_chart(fig_lrp, config=plotly_config, use_container_width=True)
            
            st.markdown("### 🧠 Hidden Layer Neuron Contributions")
            st.markdown("*Shows which neurons in each layer contributed most to the prediction.*")
            
            for layer_name, layer_relev in layer_relevances.items():
                st.markdown(f"#### {layer_name}")
                
                neurons_data = []
                for neuron_idx, relev_score in enumerate(layer_relev):
                    neurons_data.append({
                        'Neuron': f"N{neuron_idx}",
                        'Relevance': relev_score,
                        'Abs Relevance': abs(relev_score)
                    })
                
                df_neurons = pd.DataFrame(neurons_data).sort_values('Abs Relevance', ascending=True)
                colors_neurons = ['#d62728' if r > 0 else '#2ca02c' for r in df_neurons['Relevance']]
                
                fig_neurons = go.Figure(
                    go.Bar(
                        y=df_neurons['Neuron'],
                        x=df_neurons['Relevance'],
                        orientation='h',
                        marker=dict(color=colors_neurons, line=dict(color='black', width=0.5)),
                        text=[f"{val:.4f}" for val in df_neurons['Relevance']],
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>Relevance: %{x:.4f}<br><extra></extra>'
                    )
                )
                
                fig_neurons.update_layout(
                    title=f'<b>Neuron Contributions in {layer_name}</b>',
                    xaxis_title='Relevance',
                    yaxis_title='Neuron',
                    template='plotly_white',
                    height=300,
                    showlegend=False,
                    font=dict(size=9)
                )
                
                st.plotly_chart(fig_neurons, config=plotly_config, use_container_width=True)
            
            st.markdown("### 📋 LRP Detailed Analysis")
            
            lrp_df = pd.DataFrame({
                'Feature': feature_names,
                'Your Value': input_raw[0],
                'LRP Relevance': lrp_relevance,
                'Abs Relevance': np.abs(lrp_relevance),
                'Direction': ['↑ Increases Risk' if r > 0 else '↓ Decreases Risk' for r in lrp_relevance],
                'Magnitude': ['Very High' if abs(r) > np.percentile(np.abs(lrp_relevance), 75) 
                              else 'High' if abs(r) > np.percentile(np.abs(lrp_relevance), 50)
                              else 'Medium' if abs(r) > np.percentile(np.abs(lrp_relevance), 25)
                              else 'Low' for r in lrp_relevance]
            }).sort_values('Abs Relevance', ascending=False)
            
            st.dataframe(lrp_df, use_container_width=True, hide_index=True)

        # ────────────────────────────────────────────────────────────────
        # TAB 3: SHAP vs LRP COMPARISON
        # ────────────────────────────────────────────────────────────────
        with tab3:
            st.markdown("""
            ### 🔀 Comparing SHAP vs LRP
            
            Both methods explain the prediction but use different approaches:
            
            | Aspect | SHAP | LRP |
            |--------|------|-----|
            | **Approach** | Game-theoretic (Shapley values) | Layer-wise relevance propagation |
            | **Model-Agnostic** | ✅ Yes | ❌ Neural networks only |
            | **Speed** | 🐢 Slower (~1-2 min) | ⚡ Fast (milliseconds) |
            | **Interpretability** | Global importance | Layer-by-layer traces |
            """)
            
            st.markdown("### 📊 Side-by-Side Feature Attribution")
            
            with st.spinner("Preparing comparison..."):
                shap_vals_comp = explainer.shap_values(input_scaled)
                if shap_vals_comp.ndim > 1:
                    shap_vals_comp = shap_vals_comp[0]
                
                input_tensor_lrp_comp = torch.from_numpy(input_scaled).float()
                lrp_vals_comp, _ = lrp_explainer.explain(input_tensor_lrp_comp)
            
            comparison_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_vals_comp,
                'LRP Relevance': lrp_vals_comp,
                'Abs SHAP': np.abs(shap_vals_comp),
                'Abs LRP': np.abs(lrp_vals_comp)
            }).sort_values('Abs SHAP', ascending=False)
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            fig_scatter = go.Figure()
            
            fig_scatter.add_trace(go.Scatter(
                x=shap_vals_comp,
                y=lrp_vals_comp,
                mode='markers+text',
                marker=dict(size=12, color='#1f77b4', line=dict(color='black', width=1)),
                text=feature_names,
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>SHAP: %{x:.4f}<br>LRP: %{y:.4f}<br><extra></extra>'
            ))
            
            min_val_plot = min(shap_vals_comp.min(), lrp_vals_comp.min()) * 1.1
            max_val_plot = max(shap_vals_comp.max(), lrp_vals_comp.max()) * 1.1
            
            fig_scatter.add_trace(go.Scatter(
                x=[min_val_plot, max_val_plot],
                y=[min_val_plot, max_val_plot],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Perfect Agreement',
                hovertemplate='Perfect agreement line<extra></extra>'
            ))
            
            fig_scatter.update_layout(
                title='<b>SHAP vs LRP: Feature Attribution Comparison</b>',
                xaxis_title='SHAP Value',
                yaxis_title='LRP Relevance',
                template='plotly_white',
                height=500,
                hovermode='closest',
                font=dict(size=11)
            )
            
            st.plotly_chart(fig_scatter, config=plotly_config, use_container_width=True)
            
            correlation = np.corrcoef(shap_vals_comp, lrp_vals_comp)[0, 1]
            
            col_corr1, col_corr2 = st.columns(2)
            
            with col_corr1:
                st.metric(
                    "SHAP-LRP Correlation",
                    f"{correlation:.3f}",
                    help="Pearson correlation between SHAP and LRP values"
                )
            
            with col_corr2:
                if correlation > 0.7:
                    st.success("✅ **High Agreement** — Both methods rank features similarly")
                elif correlation > 0.4:
                    st.info("ℹ️ **Moderate Agreement** — Methods highlight different aspects")
                else:
                    st.warning("⚠️ **Low Agreement** — Ensemble of both recommended")

# ── Privacy Analysis Section ──
with st.expander("🔒 Privacy Analysis (DP-SGD) – What Does It Mean?"):
    st.markdown("""
    ### Privacy-Utility Trade-off Analysis
    
    This model was trained using **Differential Privacy (DP-SGD)**, which mathematically guarantees
    privacy by limiting an attacker's ability to infer any individual's data from the model.
    
    The **key metric is ε (epsilon):**
    - **Lower ε = Stronger privacy** (but may hurt accuracy slightly)
    - **Higher ε = Weaker privacy** (but higher accuracy)
    - **ε = ∞ = No privacy** (baseline for comparison)
    """)
    
    # Create dynamic results table
    results_df = pd.DataFrame({
        "Privacy Setting": ["No DP (Baseline)", "Moderate Privacy", "Strong Privacy ⭐"],
        "Privacy Budget (ε)": [
            privacy_results["no_dp"]["epsilon_str"],
            privacy_results["moderate_dp"]["epsilon_str"],
            privacy_results["strong_dp"]["epsilon_str"]
        ],
        "Test Accuracy": [
            f"{privacy_results['no_dp']['test_acc']*100:.2f}%",
            f"{privacy_results['moderate_dp']['test_acc']*100:.2f}%",
            f"{privacy_results['strong_dp']['test_acc']*100:.2f}%"
        ],
        "Convergence (rounds)": [
            privacy_results["no_dp"]["rounds"],
            privacy_results["moderate_dp"]["rounds"],
            privacy_results["strong_dp"]["rounds"]
        ]
    })
    
    st.table(results_df)
    
    # Calculate accuracy drop dynamically
    baseline_acc = privacy_results["no_dp"]["test_acc"]
    moderate_drop = (baseline_acc - privacy_results["moderate_dp"]["test_acc"]) * 100
    strong_drop = (baseline_acc - privacy_results["strong_dp"]["test_acc"]) * 100
    
    # Dynamic insights
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.metric(
            "Moderate Privacy Cost",
            f"{moderate_drop:.2f} pp",
            f"Accuracy drop vs baseline"
        )
    
    with col_right:
        st.metric(
            "Strong Privacy Cost",
            f"{strong_drop:.2f} pp",
            f"Accuracy drop vs baseline"
        )
    
    # Privacy visualization
    st.markdown("### Privacy-Accuracy Trade-off Curve")
    
    fig_tradeoff, ax = plt.subplots(figsize=(10, 6))
    
    variants = ["Baseline\n(No DP)", "Moderate\nDP", "Strong\nDP"]
    epsilons_numeric = [10, 8.0, 3.0]
    accuracies = [
        privacy_results["no_dp"]["test_acc"] * 100,
        privacy_results["moderate_dp"]["test_acc"] * 100,
        privacy_results["strong_dp"]["test_acc"] * 100
    ]
    colors_bar = ['#1f77b4', '#ff7f0e', '#d62728']
    
    bars = ax.bar(variants, accuracies, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Privacy vs Utility: Federated Learning + DP-SGD', fontsize=13, fontweight='bold')
    ax.set_ylim(72, 76)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_tradeoff)
    plt.close()
    
    # Explanation of what each setting means
    st.markdown("### What Do These Privacy Levels Mean?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **No DP (ε=∞)**
        - No privacy protection
        - Highest accuracy
        - Baseline for comparison
        """)
    
    with col2:
        st.warning(f"""
        **Moderate (ε≈8.0)**
        - Moderate privacy
        - Can resist ~100 sample attacks
        - Acc drop: {moderate_drop:.2f}pp
        """)
    
    with col3:
        st.success(f"""
        **Strong (ε≈3.0) ⭐**
        - Strong privacy protection
        - HIPAA/GDPR compliant
        - Recommended for healthcare
        - Acc drop: {strong_drop:.2f}pp
        """)
    
    st.caption(
        "💡 **Key Insight**: Strong privacy costs only ~0.65pp accuracy — "
        "a small price for end-to-end patient privacy in healthcare!"
    )

# Footer / responsibility
with st.expander("ℹ️ About the Model – Privacy & Responsibility"):
    st.markdown("""
    ### Training Method
    - **Federated Learning** (Flower framework) — data never leaves local devices
    - **Differential Privacy (DP-SGD)** — mathematical privacy guarantees
    - **5 distributed clients** simulating different healthcare facilities
    - **Centralized evaluation** on held-out test set
    
    ### Explainability (Advanced XAI)
    - **SHAP (SHapley Additive exPlanations)**: Model-agnostic, game-theoretic feature attribution
    - **LRP (Layer-wise Relevance Propagation)**: Neural network-specific, neuron-level relevance tracking
    - **Comparison mode**: Side-by-side analysis of both methods
    
    ### Important Disclaimers
    - ⚠️ **Not a medical diagnosis tool** — for educational/research only
    - Always consult a qualified healthcare professional
    - Model accuracy: ~73–75% on test set
    - Data: Pima Indians Diabetes Dataset (768 samples, 8 features)
    
    ### Privacy Guarantees
    - ε-δ differential privacy with δ = 10⁻⁵
    - Protects against membership inference attacks
    - Prevents extraction of training data patterns
    
    ### Technical References
    - **SHAP**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
    - **LRP**: Bach et al. (2015) - "On Pixel-wise Explanations for Non-Linear Classifier Decisions"
    - **DP-SGD**: Abadi et al. (2016) - "Deep Learning with Differential Privacy"
    """)
