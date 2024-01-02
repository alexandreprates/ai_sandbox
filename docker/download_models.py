from huggingface_hub import snapshot_download
import chardet

with open("/models.txt", 'r' ) as models:
    for model in models.readlines():
        print(model.strip())
        snapshot_download(model.strip())

print("Done!")