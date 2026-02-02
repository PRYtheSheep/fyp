import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout

st.set_page_config(
    layout="wide"
)

nodes = [StreamlitFlowNode(id='in', pos=(100, 100), data={'content': 'Input Tokens'}, node_type='input', source_position='right', draggable=False),]

for i in range(0,26):
    nodes.extend([
        StreamlitFlowNode(id=f'attn_{i}', pos=(250+200*i, 100), data={'content': f'ATN{i}'}, node_type='default', source_position='right', target_position='left', draggable=False),
        StreamlitFlowNode(id=f'mlp_{i}', pos=(315+200*i, 100), data={'content': f'MLP{i}'}, node_type='default', source_position='right', target_position='left', draggable=False),
        StreamlitFlowNode(id=f'rsd_{i}', pos=(395+200*i, 117), data={}, node_type='default', source_position='right', target_position='left', draggable=False, style={'width':10, 'height':10})
        ])
     
nodes.extend([
    StreamlitFlowNode(id=f'unembed', pos=(250+200*26, 100), data={'content': f'Unembedding'}, node_type='default', source_position='right', target_position='left', draggable=False),
    StreamlitFlowNode(id=f'out', pos=(5450+120, 100), data={'content': f'Output Token'}, node_type='default', source_position='right', target_position='left', draggable=False),
])

edges = [
    # StreamlitFlowEdge('in-connector_0', 'in', 'connector_0', animated=True),
    # StreamlitFlowEdge('connector_0-attn_0', 'connector_0', 'attn_0', animated=True),
    # StreamlitFlowEdge('attn_0-mlp_0', 'attn_0', 'mlp_0', animated=True),
]

if 'click_interact_state' not in st.session_state:
	st.session_state.click_interact_state = StreamlitFlowState(nodes, edges)

updated_state = streamlit_flow('ret_val_flow',
				st.session_state.click_interact_state,
				fit_view=True,
				get_node_on_click=True,
				get_edge_on_click=True)

st.write(f"Clicked on: {updated_state.selected_id}")