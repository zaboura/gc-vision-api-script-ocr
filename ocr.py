# Import libraries
from google.cloud import vision
from google.cloud import storage
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from PIL import Image
import json
import re
import matplotlib.pyplot as plt
import calendar
import spacy
import numpy as np
import os


# ! Add your credential JASON file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="path_to_credential_json_file.json"

def process_text(text):
    # preprocess the text
    # Lower case
    text = " ".join(text.lower().split())
    # remove tabulation and punctuation
    text = text.replace('[^\w\s\(\)]',' ')
    # digits
    text = text.replace('\d+', '')

    # remove stop words
    stop = stopwords.words('english')
    text = " ".join(x for x in text.split() if x not in stop)
    # return text
    return text

def async_detect_document(gcs_source_uri, gcs_destination_uri):
    # Supported mime_types are: 'application/pdf' and 'image/tiff'
    mime_type = 'application/pdf'

    # How many pages should be grouped into each json output file.
    # you can add the number of document wanted to extract text from
    batch_size = 1
    # We specify the annotation process to text detection in pdf/image
    client = vision.ImageAnnotatorClient()
    
    # Here we specify the feature type to document text detection 
    feature = vision.Feature(
        type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

    #Here, we tell the Cloud Vision API that our source type is mime_type
    #aka, a PDF, and where that PDF is found
    gcs_source = vision.GcsSource(uri=gcs_source_uri)
    input_config = vision.InputConfig(
        gcs_source=gcs_source, mime_type=mime_type)
    
    #This chunk of code says we will be creating JSON files with
    #batch_size pages worth of annotation data each, and that we will
    #put these files in gcs_destination_uri
    gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.OutputConfig(
        gcs_destination=gcs_destination, batch_size=batch_size)

    #We are making an asynchronous request using the input and output
    #configurations we just set up in the last two chunks
    async_request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config,
        output_config=output_config)
    
    #The operation we will be running will asynchronously batch-annotate files
    #using the client and asyn_request we set up earlier
    operation = client.async_batch_annotate_files(
        requests=[async_request])

    print('Waiting for the operation to finish.')
    operation.result(timeout=420)

def write_to_text(gcs_destination_uri):
    # Once the request has completed and the output has been
    # written to GCS, we can list all the output files.
    storage_client = storage.Client()
    # ! URI shoud have the format gs://bucket_name/folder_name
    match = re.match(r'gs://([^/]+)/(.+)', gcs_destination_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)

    bucket = storage_client.get_bucket(bucket_name)

    # List objects with the given prefix.
    blob_list = list(bucket.list_blobs(prefix=prefix))[1:]
    print('Output files:')

    # transcription = open("transcription.txt", "w")

    for blob in blob_list:
        print(blob.name)
    # Process the first output file from GCS.
    # Since we specified batch_size=2, the first response contains
    # the first two pages of the input file.
    for n in  range(len(blob_list)):
        output = blob_list[n]

        json_string = output.download_as_string()
        response = json.loads(json_string)


        # The actual response for the first page of the input file.
        for m in range(len(response['responses'])):

            first_page_response = response['responses'][m]

            try:
                annotation = first_page_response['fullTextAnnotation']
            except(KeyError):
                print("No annotation for this page.")

            # Here we print the full text from the first page.
            # The response contains more information:
            # annotation/pages/blocks/paragraphs/words/symbols
            # including confidence scores and bounding boxes
            # print('Full text:\n')
            # print(annotation['text'])
            # Save the text in text file
            with open("transcription.txt", "a+", encoding="utf-8") as f:
                f.write(annotation['text'])


def process_text(text):
    ## Lower case and remove digits
    text = " ".join([word for word in text.lower().split() if not word.isdigit()])
    ## remove tabulation and punctuation
    # text = text.replace('[^\w\s\(\)]',' ')
    text = re.sub(r'[^\w\s]', '', text)
    ## remove months name
    months_list = [month.lower() for month in list(calendar.month_abbr)[1:]+list(calendar.month_name)[1:]]
    text = " ".join([word for word in text.split() if word not in months_list])
    text = text.replace('\d+', '')

    #remove stop words
    stop = stopwords.words('english')
    text = " ".join(x for x in text.split() if x not in stop)

    return text



def word_cloud_plot(text):

    # Import image to np.array
    mask = np.array(Image.open('static/ai.png'))

    wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
                          background_color='black', colormap='rainbow', 
                          collocations=False, stopwords = STOPWORDS, mask=mask).generate(text)

    # plot the WordCloud image
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    # Save the image
    wordcloud.to_file("wordcloud.png")

def personal_information_filter(text):

    # We will use Part od Speech Tagging to remove unnecessary tokens
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    filtered_text = ""
    for token in doc:
        # Remove verbs, numbers, auxiliaries,...,etc
        if token.pos_ in ['VERB', 'NUM' , 'DET', 'AUX', 'SYM'] or token.ent_type in ['CARDINAL', 'ORG' , 'PERSON' , 'DATE' , 'GPE']:
            new_token = ""
        elif token.pos_ == "PUNCT":
            new_token = ""
        else:
            new_token = " {}".format(token.text)
        filtered_text += new_token
    filtered_text = filtered_text[1:]
    return filtered_text


if "__main__" == __name__:

    async_detect_document("gs://ocr-test-data/pdfs/Software-Engineering-Resume-Sample-compressed.pdf", "gs://ocr-test-data/results/")
    
    write_to_text("gs://ocr-test-data/results/")
    # Read text
    text = open("transcription.txt", "r").read()
    # Remove personal information
    text = personal_information_filter(text)
    # Process text, remove stopwords, ....
    text = process_text(text)
    # Display wordcloud of the text
    word_cloud_plot(text)
    