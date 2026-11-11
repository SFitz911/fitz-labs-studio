# ComfyUI workflows (only needed for the "My rented GPU" engine)

Build the workflow in ComfyUI on your instance, then use **Save (API format)** and drop the JSON here as:

- `image.json`      text to image
- `edit.json`       image edit (uses {{image}})
- `video.json`      text to video
- `video_i2v.json`  image to video (uses {{image}})
- `voice.json`      text to speech

Replace the values you want the studio to control with these tokens (as strings, quotes included):

| Token | Filled with |
|---|---|
| `"{{prompt}}"` | the prompt text (can also sit inside a longer string) |
| `"{{image}}"` | filename of the uploaded start/reference image (LoadImage node) |
| `"{{width}}"`, `"{{height}}"` | pixel size for the chosen aspect |
| `"{{frames}}"` | seconds x 16 + 1 |
| `"{{seed}}"` | a random seed |

The server takes the last saved output (video preferred) from the run.
