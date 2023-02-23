import re
import csv
def convert_txt_to_tsv(input_file, output_file):

    """ This function takes a txt file and seperates the sentenctes
    from . to . with tab in TSV and paragrafs n/ to n/ with tab in TSV."""

    # read in the text from the file
    with open(input_file, "r", encoding='utf-8') as f:
        text = f.read()

    # split the text into separate paragraphs
    paragraphs = re.sub(r'(?<!\n)\n(?!\n)', '', text).split("\n")

    # open the output file for writing
    with open(output_file, "w") as f:
        # loop through each paragraph and write it out with a newline
        for paragraph in paragraphs:
            # write the paragraph to the TSV file
            f.write(f"{paragraph.strip()}\n")

            ## split the paragraph into sentences
            #sentences = re.split("(?<=[.!?]) +", paragraph)

            ## loop through each sentence and write it out with a newline
            #for sentence in sentences:
            #    f.write(f"{sentence.strip()}\n")

def enumerate_tsv(input_file, output_file):
    # open the input TSV file
    with open(input_file, "r", newline='') as f_in:
        reader = csv.reader(f_in, delimiter='\t')

        # if the content is empty remove the row:
        reader = [row for row in reader if row]

        # open the output TSV file
        with open(output_file, "w", newline='') as f_out:
            writer = csv.writer(f_out, delimiter='\t')

            # loop through each row in the input file and add an enumeration column
            for i, row in enumerate(reader, 1):
                # write out the updated row with the enumeration column added
                writer.writerow([i-1] + row)

# Path: Documents/
#%%

text_rise = convert_txt_to_tsv('Documents/rise_of_universities.txt', 'Documents/book_tabbed.tsv')
text_rise = enumerate_tsv('Documents/book_tabbed.tsv', 'Documents/book_tabbed_enumerated.tsv')