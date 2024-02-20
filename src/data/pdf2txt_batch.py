import os
import glob
from pypdf import PdfReader

def convert_pdf_to_txt(pdf_path, target_directory):
    """
    Convert a PDF file to a text file and save it in the target directory.

    :param pdf_path: Path to the source PDF file.
    :param target_directory: The directory where the output text file will be saved.
    """
    base_filename = os.path.basename(pdf_path).rsplit('.', 1)[0] + '.txt'
    txt_path = os.path.join(target_directory, base_filename)

    reader = PdfReader(pdf_path)
    text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  
            text.append(page_text)

    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write('\n'.join(text))

    print(f"PDF content saved to text file: {txt_path}")

def convert_all_pdfs(source_directory, target_directory):
    os.makedirs(target_directory, exist_ok=True)

    pdf_paths = glob.glob(os.path.join(source_directory, '*.pdf'))


    for pdf_path in pdf_paths:
        convert_pdf_to_txt(pdf_path, target_directory)


source_directory = '../../data/papers' 
target_directory = '../../data/paper_txt' 
convert_all_pdfs(source_directory, target_directory)
