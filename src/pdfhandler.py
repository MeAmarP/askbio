import fitz  # PyMuPDF
import os
import json

def extract_content_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    topics = []
    image_folder = 'extracted_images'
    os.makedirs(image_folder, exist_ok=True)
    
    for page in doc:
        page_number = page.number
        text = page.get_text("text")
        topic_name = text.split('\n')[0]  # Assuming the first line is the topic name
        topic_content = '\n'.join(text.split('\n')[1:])
        tags = []  # Placeholder for tags
        
        images_info = []
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, 1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_filename = f'image_{page_number}_{img_index}.png'
            image_path = os.path.join(image_folder, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(base_image["image"])
            images_info.append(os.path.join(image_folder, image_filename))

        topics.append({
            "page_number": page_number,
            "topic_name": topic_name,
            "topic_content": topic_content,
            "tags": tags,
            "image_links": images_info
        })

    # Write to JSON
    with open('extracted_content.json', 'w') as json_file:
        json.dump(topics, json_file, indent=4)

extract_content_and_images('path_to_your_pdf_file.pdf')
