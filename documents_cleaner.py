import re
def convert_txt_to_tsv(input_file, output_file):

    """ This function takes a txt file and seperates the sentenctes
    from . to . with tab in TSV and paragrafs n/ to n/ with tab in TSV."""

    # read in the text from the file
    with open(input_file, "r", encoding='utf-8') as f:
        text = f.read()

        # split the text into paragraphs
    paragraphs = re.split("\n", text)

    # open the output file for writing
    with open(output_file, "w") as f:
        # loop through each paragraph and write it out with a newline
        for paragraph in paragraphs:
            # write the paragraph to the TSV file
            f.write(f"{paragraph.strip()}\n")

            # split the paragraph into sentences
            sentences = re.split("(?<=[.!?]) +", paragraph)

            # loop through each sentence and write it out with a newline
            for sentence in sentences:
                f.write(f"{sentence.strip()}\n")

# Path: Documents/
#%%

text_rise = convert_txt_to_tsv('Documents/rise_of_universities.txt', 'Documents/book_tabbed.tsv')
