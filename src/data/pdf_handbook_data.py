import re
from pypdf import PdfReader
from urllib.parse import urlparse, unquote
import requests

def download_pdf(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

def extract_filename_from_url(url):
    parsed_url = urlparse(url)
    path = unquote(parsed_url.path)
    filename = path.split('/')[-1]
    return filename

def read_pdf_sections(filename, pages_to_read=10):
    reader = PdfReader(filename)
    text = ""
    
    for i in range(min(pages_to_read, len(reader.pages))):
        page_text = reader.pages[i].extract_text() or ""
        text += "\n" + page_text  # Ensure there's a newline between pages
    
    # Regular expression to match section numbers (e.g., 1.1, 2.2.3)
    section_pattern = r'\n(\d+(\.\d+)+)'
    sections = []
    
    matches = list(re.finditer(section_pattern, text))
    
    for i in range(len(matches)):
        start_index = matches[i].start()
        end_index = matches[i+1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start_index:end_index].strip()
        sections.append(section_text)
    
    return sections

pdf_urls = ['https://lti.cs.cmu.edu/sites/default/files/PhD_Student_Handbook_2023-2024.pdf', 'https://lti.cs.cmu.edu/sites/default/files/MLT%20Student%20Handbook%202023%20-%202024.pdf',
"https://lti.cs.cmu.edu/sites/default/files/MIIS%20Handbook_2023%20-%202024.pdf", "https://lti.cs.cmu.edu/sites/default/files/MCDS%20Handbook%2023-24%20AY.pdf", "https://msaii.cs.cmu.edu/sites/default/files/Handbook-MSAII-2022-2023.pdf"]

pdf_filenames = [extract_filename_from_url(url) for url in pdf_urls]

for i,url in enumerate(pdf_urls):
    download_pdf(url, pdf_filenames[i])
    sections = read_pdf_sections(pdf_filenames[i], 1e8)
    with open(pdf_filenames[i][:-4]+".txt", 'w', encoding='utf-8') as file:
        for i, section_text in enumerate(sections):
            file.write(f"{section_text}\n\n")

