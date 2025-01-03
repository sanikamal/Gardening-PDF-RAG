import gradio as gr
from helpers import process_query

with gr.Blocks(theme=gr.themes.Monochrome()) as iface:
    gr.Markdown(
        """
        # 📚 DAG Scripts Q&A Assistant
        Ask any question about the DAG C1 & C2 content and get detailed answers with source references.
        """
    )

    with gr.Column():
        gr.Markdown("### Your Question")
        question_input = gr.Textbox(
            lines=3,
            placeholder="Enter your question here...",
            label="",  # Removed the label since we're using Markdown
        )
        submit_btn = gr.Button("Submit Question", variant="primary", size="lg")

    with gr.Tabs():
        with gr.TabItem("📝 Response"):
            gr.Markdown("### AI Response")
            response_output = gr.Textbox(
                lines=15,
                label="",  # Removed the label since we're using Markdown
                show_copy_button=True,
            )
        with gr.TabItem("🔍 Document References"):
            gr.Markdown("### Source Documents")
            references_output = gr.Textbox(
                lines=15,
                label="",  # Removed the label since we're using Markdown
                show_copy_button=True,
            )

    # Add submit button click event and enter key functionality
    submit_btn.click(
        fn=process_query,
        inputs=[question_input],
        outputs=[response_output, references_output],
    )
    question_input.submit(
        fn=process_query,
        inputs=[question_input],
        outputs=[response_output, references_output],
    )
	


# Close the server (in case you run this cell multiple times)
# iface.close()

# Spin up the gradio app
iface.launch(share=False)