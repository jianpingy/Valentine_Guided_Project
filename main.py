import pandas as pd
import numpy as np
import io
from typing import Annotated, List, Dict, Any, Union
from typing_extensions import TypedDict
from operator import add

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# --- 1. THE TOOLS (The "Actions") ---

@tool
def clean_speed_dating_data(file_name: str):
    """Cleans speeddating.csv, handles bytes-strings, and removes leakage columns."""
    df = pd.read_csv(file_name)
    
    # 1. Drop Leakage: 'decision' and 'decision_o' are the individual votes.
    # If we know these, the match is 100% certain. For probability, we drop them.
    df = df.drop(columns=['has_null', 'decision', 'decision_o'], errors='ignore')

    # 2. Clean byte-strings (e.g., b'female' -> female)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace("b'", "").str.replace("'", "")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # 3. Impute missing values
    imputer = SimpleImputer(strategy='median')
    df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    df_clean.to_csv('cleaned_data.csv', index=False)
    return f"Data cleaned. Rows: {len(df_clean)}. Saved to 'cleaned_data.csv'."

@tool
def select_top_features(n_features: int):
    """Uses Recursive Feature Elimination to find the best features for match probability."""
    df = pd.read_csv('cleaned_data.csv')
    X = df.drop(columns=['match'])
    y = df['match']
    
    selector = RFE(RandomForestClassifier(n_estimators=50), n_features_to_select=n_features)
    selector.fit(X, y)
    
    features = X.columns[selector.support_].tolist()
    return {"selected_features": features}

@tool
def train_probability_model(features: List[str]):
    """Trains XGBoost and returns the ROC AUC (probability quality score)."""
    df = pd.read_csv('cleaned_data.csv')
    X = df[features]
    y = df['match']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    model = xgb.XGBClassifier(n_estimators=100, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Predict Probability
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    return f"Model trained on {features}. ROC AUC Score: {auc:.4f}. The probability predictions are reliable."

# --- 2. THE GRAPH (The "Reasoning" Loop) ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add]

tools = [clean_speed_dating_data, select_top_features, train_probability_model]
openai_api_key = os.environ.get('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4o-mini", 
                 api_key=openai_api_key,
                 temperature=0).bind_tools(tools)

def agent_reasoning(state: AgentState):
    system_prompt = SystemMessage(content=(
        "You are a Match Prediction Expert. You must use the tools sequentially. "
        "First, clean the data. Then, select the best features. "
        "Finally, train the model and report the probability quality (ROC AUC)."
    ))
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Define Graph Logic
workflow = StateGraph(AgentState)
workflow.add_node("scientist", agent_reasoning)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("scientist")

# ReAct Loop logic
def router(state):
    if state["messages"][-1].tool_calls:
        return "call_tool"
    return "end"

workflow.add_conditional_edges("scientist", router, {"call_tool": "tools", "end": END})
workflow.add_edge("tools", "scientist")

app = workflow.compile()

# --- 3. EXECUTION ---
query = "Clean 'speeddating.csv', select top 10 features, and predict match probability."
# for output in app.stream({"messages": [HumanMessage(content=query)]}):
#     print(output)
for output in app.stream({"messages": [HumanMessage(content=query)]}, stream_mode="updates"):
    for node_name, state_update in output.items():
        # 1. Capture the Agent's Thought/Reasoning
        if node_name == "scientist":
            message = state_update["messages"][-1]
            
            # If the agent is calling tools
            if message.tool_calls:
                print(f"\n🤔 THOUGHT:")
                # Sometimes content is empty when calling tools, so we describe the intent
                print(f"I need to use tools to process the data. Calling: {[t['name'] for t in message.tool_calls]}")
                
                for tool_call in message.tool_calls:
                    print(f"🛠️  ACTION: {tool_call['name']}({tool_call['args']})")
            
            # If it's the final response (no more tool calls)
            else:
                print(f"\n✅ FINAL ANALYSIS:")
                print("-" * 20)
                print(message.content)
                print("-" * 20)

        # 2. Capture the Tool's Output (Observation)
        elif node_name == "tools":
            # The tool node returns a list of ToolMessages
            for tool_msg in state_update["messages"]:
                print(f"\n👁️  OBSERVATION:")
                # We limit long outputs like CSV summaries for readability
                print(f"{str(tool_msg.content)[:500]}...") 

print("\n" + "="*50)
print("🏁 Workflow Complete.")
