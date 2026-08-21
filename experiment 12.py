import os
import re
import time
import gradio as gr
from groq import Groq


# ============================================================
# 1. GROQ API KEY
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    api_key = input("Enter your Groq API key: ").strip()

if not api_key:
    raise ValueError("Groq API key was not provided.")

client = Groq(api_key=api_key)

print("Groq client initialized successfully.")


# ============================================================
# 2. RELEVANCE SCORE
# ============================================================

def calculate_relevance(prompt, response):

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "to", "of", "in", "on", "for", "and", "or",
        "with", "what", "how", "why", "write", "explain",
        "describe", "give", "about", "its", "this",
        "include", "their", "three"
    }

    prompt_words = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", prompt.lower())
    )

    response_words = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", response.lower())
    )

    important_words = prompt_words - stop_words

    if not important_words:
        return 100.0

    matched_words = important_words.intersection(response_words)

    score = (
        len(matched_words) / len(important_words)
    ) * 100

    return round(score, 2)


# ============================================================
# 3. GENERATE RESPONSE
# ============================================================

def generate_and_evaluate(prompt, temperature, max_tokens):

    if not prompt or not prompt.strip():

        return (
            "Please enter a valid prompt.",
            {
                "Status": "No prompt provided"
            }
        )

    try:

        print("\nGenerating response...")
        print("Prompt:", prompt)

        start_time = time.perf_counter()

        completion = client.chat.completions.create(

            # Current Groq production model
            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Generative AI assistant. "
                        "Provide accurate, clear and well-structured answers."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=float(temperature),

            max_tokens=int(max_tokens)
        )

        end_time = time.perf_counter()

        # Get generated response
        generated_response = (
            completion.choices[0].message.content.strip()
        )

        # Calculate response time
        latency = end_time - start_time

        # Calculate word count
        word_count = len(
            generated_response.split()
        )

        # Calculate character count
        character_count = len(
            generated_response
        )

        # Calculate relevance
        relevance_score = calculate_relevance(
            prompt,
            generated_response
        )

        # Evaluation metrics
        evaluation = {

            "Model": "llama-3.3-70b-versatile",

            "Response Time (seconds)": round(
                latency,
                3
            ),

            "Generated Word Count": word_count,

            "Generated Character Count": character_count,

            "Keyword Relevance Score (%)": relevance_score,

            "Temperature": float(temperature),

            "Maximum Tokens": int(max_tokens),

            "Status": "Successfully generated"
        }

        print("Response generated successfully.")

        return generated_response, evaluation

    except Exception as error:

        # Print error in terminal
        print("\n" + "=" * 60)
        print("ERROR WHILE GENERATING RESPONSE")
        print("=" * 60)
        print(error)
        print("=" * 60)

        # Show error in Gradio
        return (
            "Unable to generate response.\n\n"
            f"Error: {error}",

            {
                "Status": "Error",
                "Error Message": str(error)
            }
        )


# ============================================================
# 4. GRADIO INTERFACE
# ============================================================

with gr.Blocks() as application:

    gr.Markdown(
        """
        # Cloud-Based Generative AI Application

        Enter a prompt to generate content using a
        cloud-based Large Language Model and evaluate
        the generated response.
        """
    )

    with gr.Row():

        # ----------------------------------------------------
        # INPUT SECTION
        # ----------------------------------------------------

        with gr.Column():

            prompt_input = gr.Textbox(

                label="Enter Prompt",

                placeholder=(
                    "Example: Explain the applications "
                    "of Generative AI in education."
                ),

                lines=6
            )

            temperature_input = gr.Slider(

                minimum=0.0,

                maximum=1.0,

                value=0.3,

                step=0.1,

                label="Temperature"
            )

            max_tokens_input = gr.Slider(

                minimum=50,

                maximum=500,

                value=250,

                step=50,

                label="Maximum Tokens"
            )

            generate_button = gr.Button(
                "Generate and Evaluate"
            )

            clear_button = gr.ClearButton()


        # ----------------------------------------------------
        # OUTPUT SECTION
        # ----------------------------------------------------

        with gr.Column():

            response_output = gr.Textbox(

                label="Generated Response",

                lines=14
            )

            evaluation_output = gr.JSON(

                label="Evaluation Metrics"
            )


    # ========================================================
    # 5. BUTTON EVENT
    # ========================================================

    generate_button.click(

        fn=generate_and_evaluate,

        inputs=[
            prompt_input,
            temperature_input,
            max_tokens_input
        ],

        outputs=[
            response_output,
            evaluation_output
        ]
    )


# ============================================================
# 6. START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\nStarting Gradio application...")

    application.launch(
        share=True,
        debug=True
    )