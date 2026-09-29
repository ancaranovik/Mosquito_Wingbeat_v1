> ## Documentation Index
> Fetch the complete documentation index at: https://docs.edgeimpulse.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Uploader

> Sign and upload local data files or annotated datasets to Edge Impulse projects from the CLI.

The CLI uploader signs local files and uploads them to the [Ingestion API](/apis/ingestion). This is useful to upload existing data samples and entire datasets, or to migrate data between Edge Impulse projects. The uploader supports the same file types supported by the Ingestion API. See the list of [supported file types](/apis/ingestion#supported-file-types).

Data can be uploaded as unlabelled or labelled in accordance with one of the supported data annotation formats, mainly the Edge Impulse [labels](/tools/specifications/data-annotation/ei-labels) format, the Edge Impulse [structured labels](/tools/specifications/data-annotation/ei-structured-labels) format, and various [object detection](/tools/specifications/data-annotation/object-detection) formats.

<Info>
  **Additional file types or annotation formats**

  If none of these above choices are suitable for your project, you can also have a look at [custom transformation blocks](/studio/organizations/custom-blocks/custom-transformation-blocks) to parse your data samples to create a dataset supported by Edge Impulse.
</Info>

<Info>
  **Upload data from Studio**

  You can also upload data directly from Studio, see Studio [uploader](/studio/projects/data-acquisition/dataset/uploader). Go to the **Data acquisition** page, and click the 'upload' icon. You can select files or folders, the category and the label directly from here.
</Info>

## Uploading via the CLI

You can upload files via the Edge Impulse CLI via:

```bash theme={"system"}
edge-impulse-uploader path/to/a/file.wav
```

You can upload multiple files in one go via glob patterns:

```bash theme={"system"}
# Upload any wave files in a single directory
edge-impulse-uploader path/to/many/*.wav
# Another example, upload any wave in any subdirectory
edge-impulse-uploader **/*.wav
```

Or by specifying the directory:

```bash theme={"system"}
edge-impulse-uploader --directory {path}
```

The first time you'll be prompted for a server, and your login credentials (see [Edge Impulse Daemon](/tools/clis/edge-impulse-cli/serial-daemon) for more information).

### Category

Files are automatically uploaded to the `training` category, but you can override the category with the `--category` option. E.g.:

```
edge-impulse-uploader --category testing path/to/a/file.wav
```

For a full list of category options, see the [Ingestion API](/apis/ingestion) files endpoints. The endpoints follow the pattern `/api/<category>/files`.

### Labeling

A label is automatically inferred from the file name, see the [Ingestion API](/apis/ingestion). You can override this with the `--label` option. E.g.:

```
edge-impulse-uploader --label noise path/to/a/file.wav
```

### Custom labeling and metadata

When a labeling method is not provided, the labels are automatically inferred from the filename through the following regex: `^[a-zA-Z0-9\s-_]+`. For example: idle.01 will yield the label `idle`.

Thus, if you want to use labels (string values) containing float values (e.g. "0.01", "5.02", etc...), automatic labeling won't work.

To bypass this limitation, you can make a JSON file containing your dataset files' info. We also support adding metadata to your samples:

**info.labels**

```json theme={"system"}
{
    "version": 1,
    "files": [{
        "path": "path1.csv",
        "category": "split",
        "label": { "type": "label", "label": "0.5" },
        "metadata": {
            "site_collected": "Amsterdam"
        }
    }, {
        "path": "path2.csv",
        "category": "split",
        "label": { "type": "label", "label": "0.7" },
        "metadata": {
            "site_collected": "Paris"
        }
    }]
}
```

* Metadata field is optional
* To upload unlabeled data use `"label": { "type": "unlabeled" }`
* To upload multi-label data use

```json theme={"system"}
"label": {
    "type": "multi-label",
    "labels": [
        {
            "label": "noise",
            "startIndex": 0,
            "endIndex": 5000
        },
        {
            "label": "nominal_mode",
            "startIndex": 5001
            "endIndex": 60000
        },
        {
            "label": "defect",
            "startIndex": 60001
            "endIndex": 60200
        }
    ],
```

And then use:

```
edge-impulse-uploader --info-file info.labels
```

### Other dataset annotation formats

*(available since Edge Impulse CLI v1.21)*

You can upload directories of data in a range of different formats:

```bash theme={"system"}
edge-impulse-uploader --directory {path}
```

By default, we try to automatically detect your dataset annotation format from the supported ones. If we cannot detect it, the uploader will output the list of formats. You can then use:

```bash theme={"system"}
edge-impulse-uploader --directory {path} --dataset-format {name}
```

### Clearing configuration

To clear the configuration, run:

```
edge-impulse-uploader --clean
```

This resets the uploader configuration and will prompt you to log in again.

### API Key

You can use an API key to authenticate with:

```
edge-impulse-uploader --api-key ei_...
```

Note that this resets the uploader configuration and automatically configures the uploader's account and project.

### Bounding boxes

<Info>
  In July 2023, we added support for many other [object detection annotation formats](/tools/specifications/data-annotation/object-detection). Below is an example of the default Edge Impulse object detection format.
</Info>

If you want to upload data for [object detection](/studio/projects/learning-blocks/blocks/object-detection), the uploader can label the data for you as it uploads it. To do this, all you need is to create a `bounding_boxes.labels` file *in the same folder* as your image files. The contents of this file are formatted as JSON with the following structure:

```json theme={"system"}
{
    "version": 1,
    "type": "bounding-box-labels",
    "boundingBoxes": {
        "mypicture.jpg": [{
            "label": "jan",
            "x": 119,
            "y": 64,
            "width": 206,
            "height": 291
        }, {
            "label": "sami",
            "x": 377,
            "y": 270,
            "width": 158,
            "height": 165
        }]
    }
}
```

You can have multiple keys under the `boundingBoxes` object, one for each file name. If you have data in multiple folders, you can create a `bounding_boxes.labels` in each folder.

<Info>
  **You don't need to upload `bounding_boxes.labels`**

  When uploading one or more images, we check whether a labels file is present in the same folder, and automatically attach the bounding boxes to the image.

  So you can just do:

  ```
  edge-impulse-uploader yourimage.jpg
  ```

  or

  ```
  edge-impulse-uploader *
  ```
</Info>

<Info>
  **Let the Studio do the work for you!**

  Unsure about the structure of the bounding boxes file? Label some data in the studio, then export this data by selecting **Dashboard > Export**. The `bounding_boxes.labels` file will be included in the exported archive.
</Info>

### Upload data from OpenMV datasets

The uploader data in the OpenMV dataset format. Pass in the option `--format-openmv` and pass the folder of your dataset in to automatically upload data. Data is automatically split between testing and training sets. E.g.:

```
$ edge-impulse-uploader --format-openmv path/to/your-openmv-dataset
```

### Other options

* `--silent` - omits information on startup. Still prints progress information.
* `--dev` - lists development servers, use in conjunction with `--clean`.
* `--hmac-key <key>` - set the HMAC key, only used for files that need to be signed such as `wav` files.
* `--concurrency <count>` - number of files to uploaded in parallel (default: 20).
* `--progress-start-ix <index>` - when set, the progress index will start at this number. Useful to split up large uploads in multiple commands while the user still sees this as one command.
* `--progress-end-ix <index>` - when set, the progress index will end at this number. Useful to split up large uploads in multiple commands while the user still sees this as one command.
* `--progress-interval <interval>` - when set, the uploader will not print an update for every line, but every `interval` period (in ms.).
* `--allow-duplicates` - to avoid pollution of your dataset with duplicates, the hash of a file is checked before uploading against known files in your dataset. Enable this flag to skip this check.

### Uploading large datasets

When using command line wildcards to upload large datasets you may encounter an error similar to this one:

```
edge-impulse-uploader *.wav
zsh: argument list too long: edge-impulse-uploader
```

This happens if the number of `.wav` files exceeds the total number of arguments allowed for a single command on your shell. You can easily work around this shell limitation by using the `find` command to call the uploader for manageable batches of files:

```
find . -name "*.wav" -print0 | xargs -0 edge-impulse-uploader
```

You can include any necessary flags by appending them to the `xargs` portion, for example if you wish to specify a `category`:

```
find training -name "*.wav" -print0 | xargs -0 edge-impulse-uploader --category training
```
