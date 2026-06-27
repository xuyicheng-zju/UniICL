"""Public release module documentation."""
import os
from PIL import Image


def build_icl_input(demos, image_dir, target_image_path, target_question):
    """Public release documentation."""
    input_list = []


    for demo in demos:
        demo_img_path = os.path.join(image_dir, demo['image_name'])
        if os.path.exists(demo_img_path):
            demo_img = Image.open(demo_img_path).convert("RGB")
            demo_question = demo.get('instruction', demo.get('text', ''))
            demo_answer = demo.get('answer', demo.get('annotation', ''))



            input_list.append("\nUser: ")
            input_list.append(demo_img)
            input_list.append(demo_question)
            input_list.append(f"\nAssistant: {demo_answer}")


    target_img = Image.open(target_image_path).convert("RGB")

    if len(demos) > 0:
        input_list.append("\nUser: ")
    else:
        input_list.append("User: ")
    input_list.append(target_img)
    input_list.append(target_question)
    input_list.append("\nAssistant:")

    return input_list
