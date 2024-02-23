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


def convert_pdf_to_txt(pdf_path, txt_path=None):
    """
    Convert a PDF file to a text file.

    :param pdf_path: Path to the source PDF file.
    :param txt_path: Path to the output text file. If None, generates a txt file name based on pdf_path.
    """
    if txt_path is None:
        txt_path = pdf_path.rsplit('.', 1)[0] + '.txt'

    # Initialize a PDF reader object and read the PDF
    reader = PdfReader(pdf_path)
    text = []

    # Extract text from each page
    for page in reader.pages:
        text.append(page.extract_text())

    # Combine all text into one string and save to the txt_path
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write('\n'.join(filter(None, text)))

    print(f"PDF content saved to text file: {txt_path}")


pdf_urls = ["https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_spring.pdf", "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_summer_1.pdf",
"https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_summer_2.pdf", "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_fall.pdf","https://www.cmu.edu/hub/calendar/docs/2324-academic-calendar-list-view.pdf"
,"https://www.cmu.edu/hub/calendar/docs/2324-doctoral-academic-calendar-list-view.pdf","https://www.cmu.edu/hub/calendar/docs/2425-academic-calendar-list-view.pdf"]

pdf_filenames = [extract_filename_from_url(url) for url in pdf_urls]

for i,url in enumerate(pdf_urls):
    download_pdf(url, pdf_filenames[i])
    convert_pdf_to_txt(pdf_filenames[i])