
def txtCleaner_tab(path: str ) -> str :

    """ This function cleans the text and makes it TSV. """

        with open(path, 'r') as file:
            text = file.read()
            text = text.replace('')

        return text

# Path: Documents/

