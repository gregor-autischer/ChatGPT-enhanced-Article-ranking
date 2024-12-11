#!/usr/bin/env python3

import sys

import openai
from openai import OpenAI

def main():
    # Check if search term is passed
    if len(sys.argv) > 1:
        search_term_string = "".join(sys.argv[1:])
    else:
        print("No Search term passed!")
        return
    
    print("Search Term:", search_term_string)

    # extend search term with ChatGPT
    additional_words_string = extend_search_term_with_gpt(search_term_string)

    print("Additional Words from ChatGPT:")
    print(additional_words_string)

    search_term = search_term_string.split()
    search_term_extended = additional_words_string.split()

    print(search_term)
    print(search_term_extended)

    # TODO: Maybe? filter out filler words. I dont know if required?

    # TODO: use search_term and search_term_extended to rank articles

def extend_search_term_with_gpt(search_term_string):
    # ChatGPT integration

    #gpt_system_content = "You are a helpful assistant."
    gpt_system_content = "You find additional synonymes and words to improve an article search"
    gpt_user_content = f"Give me additional words and synonyms that improve a keyword search for {search_term_string}. Just return between 5 and 15 words as a string with no bulletpoints or similar. Return only words no phrases."

    try:
        client = OpenAI()
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {"role": "system", "content": gpt_system_content},
                {"role": "user", "content": gpt_user_content}
            ]
        )
        additional_words_string = completion.choices[0].message.content
    
    except openai.OpenAIError as e:
        print(f"An OpenAI error occurred: {str(e)}")
        sys.exit("Stopping the program due to error.")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit("Stopping the program due to error.")

    return additional_words_string



if __name__ == "__main__":
    main()