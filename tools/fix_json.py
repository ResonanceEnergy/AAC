#!/usr/bin/env python3
"""
Fix the malformed JSON in sample_reddit_data.json
"""

import json

def fix_json_file():
    with open('sample_reddit_data.json', 'r') as f:
        content = f.read()

    print("Original content around error:")
    start = 4270
    end = 4290
    print(content[start:end])

    # The issue is that "ossposts" should be "num_crossposts"
    # and there might be missing properties
    content = content.replace('"ossposts":', '"num_crossposts":')

    # Try to add missing properties if needed
    # Look for the pattern where the object closes too early
    content = content.replace('"is_video": false}},', '"is_video": false, "post_hint": null}},')

    with open('sample_reddit_data.json', 'w') as f:
        f.write(content)

    print("Fixed JSON file")

    # Test if it's valid now
    try:
        with open('sample_reddit_data.json', 'r') as f:
            data = json.load(f)
        print("JSON is now valid!")
        return True
    except json.JSONDecodeError as e:
        print(f"Still has error: {e}")
        return False

if __name__ == "__main__":
    fix_json_file()