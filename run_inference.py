import logging
import os
from Layoutlmv3_inference.ocr import prepare_batch_for_inference
from Layoutlmv3_inference.inference_handler import handle

def main(model_path, image_path):
    try:
        # Prepare the batch for a single image
        inference_batch = prepare_batch_for_inference([image_path])
        print(f'inference batch : {inference_batch}')

        # Create a context dictionary with the model directory
        context = {"model_dir": model_path}
        # print(f'context : {context}')

        # Perform inference
        handle(inference_batch, context)
        print("Inference completed on image:", image_path)

    except Exception as err:
        os.makedirs('log', exist_ok=True)
        logging.basicConfig(filename='log/error_output.log', level=logging.ERROR,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s')
        logger = logging.getLogger(__name__)
        logger.error(err)
        print(f"Error: {err}")

if __name__ == "__main__":
    # Path to the model directory
    model_path = './layoutlmv3-finetuned-ubiai-dataset-invoice-3/final_model'  # Adjust this to your local model path

    # Path to a single image in Google Drive
    image_path = './test_images/page_1.png'  # Change this to the path of your image

    # Run the main function
    main(model_path, image_path)
