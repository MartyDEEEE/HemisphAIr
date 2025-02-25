import requests
import json
import sys
import argparse
import time

def stream_query_llm(model_name, prompt, max_tokens=200):
    """
    Send a prompt to an LLM via Ollama API and stream the response in real-time.
    
    Args:
        model_name (str): Name of the LLM model to use
        prompt (str): The prompt to send to the LLM
        max_tokens (int): Maximum number of tokens in the response
        
    Returns:
        str: The complete LLM's response
    """
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True
    }
    
    response = requests.post(url, json=data, stream=True)
    full_response = ""
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    sys.stdout.write(chunk["response"])
                    sys.stdout.flush()
                    full_response += chunk["response"]
                if chunk.get("done", False):
                    break
        print()  # Add newline after response
        return full_response
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return "Error communicating with the LLM."

def hemispheric_brain_simulation(llm1_model, llm2_model, user_query, max_turns=6, max_tokens_per_turn=200):
    """
    Simulate a hemispheric brain model with two LLMs communicating with each other.
    
    Args:
        llm1_model (str): Name of the first LLM model
        llm2_model (str): Name of the second LLM model
        user_query (str): The initial query from the user
        max_turns (int): Maximum number of turns for each LLM
        max_tokens_per_turn (int): Maximum tokens per response
        
    Returns:
        str: The final summary from LLM1
    """
    print(f"\n🧠 Hemispheric Brain Simulation")
    print(f"📝 Query: {user_query}\n")
    print(f"🔄 Starting conversation between {llm1_model} and {llm2_model}...\n")
    
    # Initial prompt to relay to LLM2
    llm1_to_llm2_prompt = f"You are participating in a hemispheric brain simulation. A user has asked: '{user_query}'. Please provide your initial thoughts on this query. Be insightful but concise."
    
    conversation_history = []
    
    # Initial response from LLM2
    print(f"🧠 {llm2_model}: ", end="")
    llm2_response = stream_query_llm(llm2_model, llm1_to_llm2_prompt, max_tokens_per_turn)
    conversation_history.append(f"{llm2_model}: {llm2_response}")
    
    # Conversation loop
    for turn in range(max_turns):
        # LLM1's turn
        llm1_prompt = f"You are participating in a hemispheric brain simulation discussing: '{user_query}'. The other hemisphere said: '{llm2_response}'. Respond with your thoughts, building on or challenging what was said."
        print(f"🧠 {llm1_model}: ", end="")
        llm1_response = stream_query_llm(llm1_model, llm1_prompt, max_tokens_per_turn)
        conversation_history.append(f"{llm1_model}: {llm1_response}")
        
        # Break if we've reached max turns
        if turn == max_turns - 1:
            break
            
        # LLM2's turn
        llm2_prompt = f"You are participating in a hemispheric brain simulation discussing: '{user_query}'. The other hemisphere said: '{llm1_response}'. Respond with your thoughts, building on or challenging what was said."
        print(f"🧠 {llm2_model}: ", end="")
        llm2_response = stream_query_llm(llm2_model, llm2_prompt, max_tokens_per_turn)
        conversation_history.append(f"{llm2_model}: {llm2_response}")
    
    # Final summary from LLM1
    summary_prompt = f"You are the summarizing hemisphere of the brain. You've had a discussion with another hemisphere about: '{user_query}'. Here's the full conversation:\n\n" + "\n\n".join(conversation_history) + "\n\nPlease provide a comprehensive and insightful summary of the discussion, highlighting key insights, areas of agreement, and any interesting perspectives that emerged."
    
    print("\n🔄 Generating final summary...\n")
    print(f"🧠 Final Summary ({llm1_model}): ", end="")
    summary = stream_query_llm(llm1_model, summary_prompt, max_tokens=400)
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Hemispheric Brain Simulation using two LLMs")
    parser.add_argument("--llm1", type=str, default="llama3", help="Name of the first LLM model")
    parser.add_argument("--llm2", type=str, default="mistral", help="Name of the second LLM model")
    parser.add_argument("--max-turns", type=int, default=6, help="Maximum number of turns for each LLM")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens per response")
    args = parser.parse_args()
    
    print(f"\n🧠 Hemispheric Brain Simulation")
    print(f"🤖 Using models: {args.llm1} and {args.llm2}")
    print(f"⚙️ Max turns: {args.max_turns}, Max tokens per turn: {args.max_tokens}")
    
    try:
        while True:
            user_query = input("\n❓ Enter your query (or 'quit' to exit): ")
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Exiting the simulation. Goodbye!")
                break
                
            hemispheric_brain_simulation(
                args.llm1, 
                args.llm2, 
                user_query, 
                max_turns=args.max_turns, 
                max_tokens_per_turn=args.max_tokens
            )
            
    except KeyboardInterrupt:
        print("\nExiting the simulation. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Make sure Ollama is running and the specified models are available.")

if __name__ == "__main__":
    main()
