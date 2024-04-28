import fitz  # PyMuPDF
import os
import json


def save_page_ranges(source_pdf_path, output_pdf_path, page_ranges):
    """
    Saves specified ranges of pages from a source PDF to a new PDF file.

    Args:
    source_pdf_path (str): Path to the source PDF file.
    output_pdf_path (str): Path to the output PDF file.
    page_ranges (list of tuples): List of tuples, where each tuple represents a page range to save (inclusive, 0-indexed).
    """
    # Open the source PDF file
    doc = fitz.open(source_pdf_path)
    # Create a new PDF to save selected pages
    new_doc = fitz.open()

    # Iterate through each range and add the pages to the new document
    for start, end in page_ranges:
        new_doc.insert_pdf(doc, from_page=start, to_page=end)

    # Save the new document
    new_doc.save(output_pdf_path)
    new_doc.close()
    doc.close()
    print(f"Specified page ranges have been saved to {output_pdf_path}")

def extract_content_and_images(pdf_path):
    """
    Extracts content from each page in the PDF and images from each page.

    Args:
        pdf_path (str): Path to the source PDF file.
    
    Returns:
        None, but writes a JSON file named 'extracted_content.json' with extracted information including page number, topic name, content, tags, and image links. 
    """
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
