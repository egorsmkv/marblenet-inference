from os.path import basename
from zipfile import ZipFile

import gradio as gr
from inference import inference_file


def extract_vad_chunks(file):
    archive_name = "tmp.zip"
    n_files = 0

    with ZipFile(archive_name, "w") as zip_file:
        results = inference_file(file)

        for result in results:
            print(result)

            arc_name = basename(result["filename"])
            zip_file.write(result["filename"], arc_name)

            n_files += 1

    gr.Success(f"VAD model identified {n_files} files")

    return archive_name


demo = gr.Interface(
    title="MarbleNet inference",
    fn=extract_vad_chunks,
    inputs=gr.File(file_count="single", file_types=[".wav"]),
    outputs="file",
    submit_btn="Inference",
)
demo.launch(share=True)
