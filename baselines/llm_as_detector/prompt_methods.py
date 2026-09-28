"""Prompting strategies for the LLM-as-detector baselines.

Adopted without modification to the prompts from the official Who&When
implementation, so the baseline numbers stay comparable. Every public function
writes one ``Prediction for <file>:`` block per trajectory to stdout; callers
redirect stdout to a log file that ``evaluate.py`` then parses.

``use_ground_truth`` selects the benchmark's two conditions: ``True`` is
``w/ GT`` (the task's reference answer is shown to the judge) and ``False`` is
``w/o GT`` (the agent logs alone).
"""

import os
import json
import random

from collections import deque

from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

#: Who&When names the acting agent under a different key per subset.
AGENT_KEY = {"Algorithm-Generated": "name", "Hand-Crafted": "role"}

DEFAULT_ENCODER = os.environ.get(
    "MASC_SENTENCE_ENCODER", "sentence-transformers/all-MiniLM-L6-v2"
)

#: Binary Search breaks an ambiguous judge reply by picking a half at random.
#: Seeding it keeps a run reproducible; override with $MASC_SEED.
_RNG = random.Random(int(os.environ.get("MASC_SEED", "42")))


# --- Helper Functions ---
def _get_sorted_json_files(directory_path):
    """Gets and sorts JSON files numerically from a directory."""
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return []
    except Exception as e:
        print(f"Error reading or sorting files in {directory_path}: {e}")
        return []

def _load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def _make_api_call(client, model, messages, max_tokens):
    """Makes an API call to Azure OpenAI."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None

# --- All-at-Once Method ---

def all_at_once(client: AzureOpenAI, directory_path: str, is_handcrafted: bool,
                model: str, max_tokens: int, use_ground_truth: bool = False):
    """
    Analyzes chat history by feeding the entire conversation at once to the model.
    """
    print("\n--- Starting All-at-Once Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = (
        AGENT_KEY["Algorithm-Generated"]
        if "Algorithm-Generated" in directory_path
        else AGENT_KEY["Hand-Crafted"]
    )

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed for evaluation

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" for entry in chat_history
        ])
        if use_ground_truth:
            prompt = (
                "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
                f"The problem is:  {problem}\n"
                f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used in prompt
                "Identify which agent made an error, at which step, and explain the reason for the error. "
                "Here's the conversation:\n\n" + chat_content +
                "\n\nBased on this conversation, please predict the following:\n"
                "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
                "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
                """
                {
                    "agent a": "xx",
                    "agent b": "xxxx",
                    "agent c": "xxxxx",
                    "agent a": "xxxxxxx"
                },
                """
                "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
                "3. The reason for your prediction."
                "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
            )
        else:
            prompt = (
                    "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
                    f"The problem is:  {problem}\n"
                    "Identify which agent made an error, at which step, and explain the reason for the error. "
                    "Here's the conversation:\n\n" + chat_content +
                    "\n\nBased on this conversation, please predict the following:\n"
                    "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
                    "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
                    """
                    {
                        "agent a": "xx",
                        "agent b": "xxxx",
                        "agent c": "xxxxx",
                        "agent a": "xxxxxxx"
                    },
                    """
                    "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
                    "3. The reason for your prediction."
                    "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
            )
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
            {"role": "user", "content": prompt},
        ]

        result = _make_api_call(client, model, messages, max_tokens)

        print(f"Prediction for {json_file}:")
        if result:
            print(result)
        else:
            print("Failed to get prediction.")
        print("\n" + "="*50 + "\n")


# --- Step-by-Step Method ---

def step_by_step(client: AzureOpenAI, directory_path: str, is_handcrafted: bool,
                 model: str, max_tokens: int, use_ground_truth: bool = False):
    """
    Analyzes chat history step by step, asking the model at each step if an error occurred.
    """
    print("\n--- Starting Step-by-Step Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = (
        AGENT_KEY["Algorithm-Generated"]
        if "Algorithm-Generated" in directory_path
        else AGENT_KEY["Hand-Crafted"]
    )
    #index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        current_conversation_history = ""
        error_found = False
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"
            if use_ground_truth:
                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                    f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used
                    f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                )
            else:
                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. " # Included as per original code - remove if ground truth shouldn't be used
                    f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                )
            messages=[
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"Evaluating Step {idx} by {agent_name}...")
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True # Treat API error as unable to proceed
                break

            print(f"LLM Evaluation: {answer}")

            # Basic check for "Yes" at the beginning of the response
            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                error_found = True
                break # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no"):
                 print("No significant error detected in this step.")
            else:
                print("Warning: Unexpected response format from LLM. Continuing evaluation.")
                # Optionally handle unexpected format more robustly

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

        print("\n" + "="*50 + "\n")


def my_step_by_step(client: AzureOpenAI, directory_path: str, is_handcrafted: bool,
                    model: str, max_tokens: int, use_ground_truth: bool = False):
    """
    Analyzes chat history step by step, asking the model at each step if an error occurred.
    """
    print("\n--- Starting Step-by-Step Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = (
        AGENT_KEY["Algorithm-Generated"]
        if "Algorithm-Generated" in directory_path
        else AGENT_KEY["Hand-Crafted"]
    )
    #index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        current_conversation_history = ""
        error_found = False
        for idx, entry in enumerate(chat_history):
            if idx == int(data["mistake_step"]):
                break
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"
            if use_ground_truth:
                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                    f"The Answer for the problem is: {ground_truth}\n" # Included as per original code - remove if ground truth shouldn't be used
                    f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                )
            else:
                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. " # Included as per original code - remove if ground truth shouldn't be used
                    f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                )
            messages=[
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"Evaluating Step {idx} by {agent_name}...")
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True # Treat API error as unable to proceed
                break

            print(f"LLM Evaluation: {answer}")

            # Basic check for "Yes" at the beginning of the response
            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                # error_found = True
                # break # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no"):
                 print("No significant error detected in this step.")
            else:
                print("Warning: Unexpected response format from LLM. Continuing evaluation.")
                # Optionally handle unexpected format more robustly

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

        print("\n" + "="*50 + "\n")

def evaluate_chat_with_sliding_window(client: AzureOpenAI, directory_path: str,
                                      is_handcrafted: bool, model: str, max_tokens: int,
                                      K: int, use_ground_truth: bool = False):
    """
    Args:
        chat_history: list of dicts, each with keys like index_agent, content
        index_agent: str, key to extract agent name
        problem: str, description of the problem being solved
        ground_truth: str, the answer to the problem
        client, model, max_tokens: used in _make_api_call
        K: int, sliding window size
    """
    # Initialize a sliding window buffer with fixed length K

    window_buffer = deque(maxlen=K)
    print("\n--- Starting Step-by-Step Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = (
        AGENT_KEY["Algorithm-Generated"]
        if "Algorithm-Generated" in directory_path
        else AGENT_KEY["Hand-Crafted"]
    )
    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")  # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")

        error_found = False
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')

            # Add current step into the sliding window
            window_buffer.append((idx, agent_name, content))

            # Build conversation history from buffer
            current_conversation_history = ""
            for step_idx, name, step_content in window_buffer:
                current_conversation_history += f"Step {step_idx} - {name}: {step_content}\n"
            if idx==0:

                start_idx = 0
                related_idx = 0
            else:
                start_idx, s_agent_name, s_content = window_buffer[0]
                related_idx, e_agent_name, e_content = window_buffer[len(window_buffer)-1]
            # Construct prompt with current window
            # prompt = (
            #     f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. "
            #     f"The problem being addressed is: {problem}. "
            #     f"The Answer for the problem is: {ground_truth}\n"
            #     f"Here is the conversation history from step ({start_idx}) to step({related_idx}):\n{current_conversation_history}\n"
            #     f"The most recent step ({idx}) was by '{agent_name}'.\n"
            #     "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
            #     "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
            #     "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
            #     "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            #
            # )
            if use_ground_truth:
                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                    f"The Answer for the problem is: {ground_truth}\n"  # Included as per original code - remove if ground truth shouldn't be used
                    f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                )
            else:
                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                    f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                )
            messages = [
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"Evaluating Step {idx} by {agent_name}...")
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True  # Treat API error as unable to proceed
                break

            print(f"LLM Evaluation: {answer}")

            # Basic check for "Yes" at the beginning of the response
            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                error_found = True
                break  # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no"):
                print("No significant error detected in this step.")
            else:
                print("Warning: Unexpected response format from LLM. Continuing evaluation.")
                # Optionally handle unexpected format more robustly

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

        print("\n" + "=" * 50 + "\n")


def evaluate_chat_with_single(client: AzureOpenAI, directory_path: str,
                              is_handcrafted: bool, model: str, max_tokens: int,
                              K: int, use_ground_truth: bool = False):
    """
    Args:
        chat_history: list of dicts, each with keys like index_agent, content
        index_agent: str, key to extract agent name
        problem: str, description of the problem being solved
        ground_truth: str, the answer to the problem
        client, model, max_tokens: used in _make_api_call
        K: int, sliding window size
    """
    # Initialize a sliding window buffer with fixed length K

    # window_buffer = deque(maxlen=K)
    print("\n--- Starting Step-by-Step Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = (
        AGENT_KEY["Algorithm-Generated"]
        if "Algorithm-Generated" in directory_path
        else AGENT_KEY["Hand-Crafted"]
    )
    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")  # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        current_conversation_history = ""
        error_found = False
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            # Add current step into the sliding window
            # Build conversation history from buffer
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"
            # for step_idx, name, step_content in window_buffer:
            #     current_conversation_history += f"Step {step_idx} - {name}: {step_content}\n"
            # if idx==0:
            #
            #     start_idx = 0
            #     related_idx = 0
            # else:
            #     start_idx, s_agent_name, s_content = window_buffer[0]
            #     related_idx, e_agent_name, e_content = window_buffer[len(window_buffer)-1]
            # Construct prompt with current window
            # prompt = (
            #     f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. "
            #     f"The problem being addressed is: {problem}. "
            #     f"The Answer for the problem is: {ground_truth}\n"
            #     f"The current step ({idx}) was performed by '{agent_name}', and the content of this step is:\n{content}\n"
            #     "Your task is to determine whether this step contains an error that could hinder the problem-solving process or lead to an incorrect solution.\n"
            #     "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
            #     "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
            #     "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            # )
            if use_ground_truth:
                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                    f"The Answer for the problem is: {ground_truth}\n"  # Included as per original code - remove if ground truth shouldn't be used
                    f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                )
            else:
                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                    f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                )
            messages = [
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"Evaluating Step {idx} by {agent_name}...")
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True  # Treat API error as unable to proceed
                break

            print(f"LLM Evaluation: {answer}")

            # Basic check for "Yes" at the beginning of the response
            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                error_found = True
                break  # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no"):
                print("No significant error detected in this step.")
            else:
                print("Warning: Unexpected response format from LLM. Continuing evaluation.")
                # Optionally handle unexpected format more robustly

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

        print("\n" + "=" * 50 + "\n")

# --- Binary Search Method ---

def _construct_binary_search_prompt(problem, answer, chat_segment_content,
                                    range_description, upper_half_desc, lower_half_desc,
                                    use_ground_truth=False):
    """Constructs the prompt for the binary search step."""
    if use_ground_truth:
        return (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
            "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
            f"The problem to address is as follows: {problem}\n"
            f"The Answer for the problem is: {answer}\n" # Included as per original code - remove if ground truth shouldn't be used
            f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
            f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
            "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
        )
    else:
        return (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
            "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
            f"The problem to address is as follows: {problem}\n"
            f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
            f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
            "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
        )
def _report_binary_search_error(chat_history, step, json_file, is_handcrafted, directory_path, use_ground_truth):
    """Reports the identified error step from binary search."""
    index_agent = (
        AGENT_KEY["Algorithm-Generated"]
        if "Algorithm-Generated" in directory_path
        else AGENT_KEY["Hand-Crafted"]
    )
    entry = chat_history[step]
    agent_name = entry.get(index_agent, 'Unknown Agent')

    print(f"\nPrediction for {json_file}:")
    print(f"Agent Name: {agent_name}")
    print(f"Step Number: {step}")
    print("\n" + "="*50 + "\n")

def _find_error_in_segment_recursive(client: AzureOpenAI, model: str, max_tokens: int,
                                     chat_history: list, problem: str, answer: str,
                                     start: int, end: int, json_file: str,
                                     is_handcrafted: bool, directory_path: str,
                                     use_ground_truth: bool = False):
    """Recursive helper function for binary search analysis."""
    if start > end:
         print(f"Warning: Invalid range in binary search for {json_file} (start={start}, end={end}). Reporting last valid step.")
         _report_binary_search_error(chat_history, end if end >= 0 else 0, json_file, is_handcrafted, directory_path, use_ground_truth) # Report something reasonable
         return
    if start == end:
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted, directory_path, use_ground_truth)
        return

    index_agent = (
        AGENT_KEY["Algorithm-Generated"]
        if "Algorithm-Generated" in directory_path
        else AGENT_KEY["Hand-Crafted"]
    )

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        print(f"Warning: Empty segment in binary search for {json_file} (start={start}, end={end}). Cannot proceed.")
        _report_binary_search_error(chat_history, start, json_file, is_handcrafted, directory_path, use_ground_truth)
        return

    chat_content = "\n".join([
        f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in segment_history
    ])

    mid = start + (end - start) // 2 

    range_description = f"from step {start} to step {end}"
    upper_half_desc = f"from step {start} to step {mid}"
    lower_half_desc = f"from step {mid + 1} to step {end}"

    prompt = _construct_binary_search_prompt(
        problem, answer, chat_content, range_description, upper_half_desc,
        lower_half_desc, use_ground_truth=use_ground_truth,
    )

    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
        {"role": "user", "content": prompt}
    ]

    print(f"Analyzing step {start}-{end} for {json_file}...")
    result = _make_api_call(client, model, messages, max_tokens)

    if not result:
        print(f"API call failed for segment {start}-{end}. Stopping binary search for {json_file}.")
        return

    print(f"LLM Prediction for segment {start}-{end}: {result}")
    result_lower = result.lower() 

    if "upper half" in result_lower:
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted, directory_path, use_ground_truth)
    elif "lower half" in result_lower:
         new_start = min(mid + 1, end)
         _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted, directory_path, use_ground_truth)
    else:
        print(f"Warning: Ambiguous response '{result}' from LLM for segment {start}-{end}. Randomly choosing a half.")
        if _RNG.randint(0, 1) == 0:
            print("Randomly chose upper half.")
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, json_file, is_handcrafted, directory_path, use_ground_truth)
        else:
            print("Randomly chose lower half.")
            new_start = min(mid + 1, end)
            _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, json_file, is_handcrafted, directory_path, use_ground_truth)


def binary_search(client: AzureOpenAI, directory_path: str, is_handcrafted: bool,
                  model: str, max_tokens: int, use_ground_truth: bool = False):
    """
    Analyzes chat history using a binary search approach to find the error step.
    """
    print("\n--- Starting Binary Search Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)

    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "") # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, 0, len(chat_history) - 1, json_file, is_handcrafted, directory_path, use_ground_truth)

def evaluate_multiagent_conversation(
        client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str,
        max_tokens: int, K: int, use_ground_truth: bool = False
):
    """
    Perform step-by-step error detection in a multi-agent conversation using a semantic sliding window.

    If current step index < window_size, all history so far is directly included in the window.
    Otherwise, a semantic similarity-based replacement strategy is used.
    """
    window_size = K
    print("\n--- Starting Step-by-Step Analysis ---\n")
    json_files = _get_sorted_json_files(directory_path)
    st_model = SentenceTransformer(DEFAULT_ENCODER)
    index_agent = (
        AGENT_KEY["Algorithm-Generated"]
        if "Algorithm-Generated" in directory_path
        else AGENT_KEY["Hand-Crafted"]
    )
    for json_file in tqdm(json_files):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")  # Keep ground truth if needed

        if not chat_history:
            print(f"Skipping {json_file}: No chat history found.")
            continue

        print(f"--- Analyzing File: {json_file} ---")
        error_found = False
        global_buffer = []       # [{'idx', 'agent', 'content', 'embedding'}]
        sliding_window = []      # Top-K semantically relevant steps

        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')
            content = entry.get('content', '')
            step_text = f"Step {idx} - {agent_name}: {content}"
            current_embedding = st_model.encode(step_text, convert_to_tensor=True)

            # Save to buffer
            current_record = {
                'idx': idx,
                'agent': agent_name,
                'content': content,
                'embedding': current_embedding,
            }
            global_buffer.append(current_record)

            # === Determine window ===
            if idx < window_size:
                # Early stage: Use all available history
                sliding_window = global_buffer[:idx + 1]
            else:
                # Full buffer: semantic similarity-based selection
                # 1. Similarity with current window
                similarities = []
                for item in sliding_window:
                    sim = util.cos_sim(current_embedding, item['embedding']).item()
                    similarities.append((sim, item))

                # 2. Remove low-similarity items
                if similarities:
                    avg_sim = sum(sim for sim, _ in similarities) / len(similarities)
                    sliding_window = [item for sim, item in similarities if sim >= avg_sim]

                # 3. Add more relevant items from buffer
                current_ids = {item['idx'] for item in sliding_window}
                all_sims = []
                for item in global_buffer:
                    sim = util.cos_sim(current_embedding, item['embedding']).item()
                    all_sims.append((sim, item))
                all_sims.sort(reverse=True, key=lambda x: x[0])

                for sim, item in all_sims:
                    if item['idx'] not in current_ids:
                        sliding_window.append(item)
                        current_ids.add(item['idx'])
                    if len(sliding_window) >= window_size:
                        break

            # === Construct prompt ===
            sorted_window = sorted(sliding_window, key=lambda x: x['idx'])
            conversation_context = ""
            for item in sorted_window:
                conversation_context += f"Step {item['idx']} - {item['agent']}: {item['content']}\n"
            if use_ground_truth:
                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                    f"The Answer for the problem is: {ground_truth}\n"  # Included as per original code - remove if ground truth shouldn't be used
                    f"Here is the conversation history up to the current step:\n{conversation_context}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                )
            else:
                prompt = (
                    f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                    f"Here is the conversation history up to the current step:\n{conversation_context}\n"
                    f"The most recent step ({idx}) was by '{agent_name}'.\n"
                    "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                    "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                    "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                    "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
                )
            messages = [
                {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                {"role": "user", "content": prompt},
            ]

            print(f"\n🔍 Evaluating Step {idx} by {agent_name}...")
            answer = _make_api_call(client, model, messages, max_tokens)

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True  # Treat API error as unable to proceed
                break

            print(f"LLM Evaluation: {answer}")

            # Basic check for "Yes" at the beginning of the response
            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                print(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                error_found = True
                break  # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no"):
                print("No significant error detected in this step.")
            else:
                print("Warning: Unexpected response format from LLM. Continuing evaluation.")
                # Optionally handle unexpected format more robustly

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

        print("\n" + "=" * 50 + "\n")
