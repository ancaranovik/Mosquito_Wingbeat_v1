> ## Documentation Index
> Fetch the complete documentation index at: https://docs.edgeimpulse.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Metadata

> Use metadata keys to organize samples, analyze distributions, and control training and validation splits.

You can add arbitrary metadata to data items. You can use this for example to track on which site data was collected, where data was imported from, or where the machine that generated the data was placed. Some key use cases for metadata are:

1. Prevent leaking data between your train and validation set. See: [Using metadata to control your train/validation split](/studio/projects/data-acquisition/dataset/metadata#using-metadata-to-control-your-train-validation-split) below.
2. Synchronisation actions in [data pipelines](/studio/organizations/data-pipelines), for example to remove data in a project if the source data was deleted in the cloud.
3. Get a better understanding of real-world accuracy by seeing how well your model performs when grouped by a metadata key. E.g. whether data on site A performs better than site B.

### Viewing and editing metadata in the Studio

Metadata is shown on **Data acquisition** when you click on a data item. From here you can add, edit and remove metadata keys.

<Frame caption="Metadata shown on Data acquisition">
  <img src="https://mintcdn.com/edgeimpulse/ZGbvQAs-QNgkltKA/.assets/images/viewing-editing-metadata.png?fit=max&auto=format&n=ZGbvQAs-QNgkltKA&q=85&s=64dd57e57e07af6c28b7b3866ed15096" alt="Data acquisition sample details panel with editable metadata keys" width="1587" height="1000" data-path=".assets/images/viewing-editing-metadata.png" />
</Frame>

### Metadata analytics

Once your data items have metadata attached, you can get a high-level picture of the composition of your dataset at a glance using the **Data distribution** panel. Click the chart icon next to the **Data Collected** stat on the **Data acquisition** page to open it.

<Frame caption="Data distribution panel: single metadata key view">
  <img src="https://mintcdn.com/edgeimpulse/4wEDaLSprog45X_b/.assets/images/metadata-data-distribution.png?fit=max&auto=format&n=4wEDaLSprog45X_b&q=85&s=ee42f70087080a372062e8619d962b00" alt="Data distribution panel charting samples for a selected metadata key" width="1549" height="1276" data-path=".assets/images/metadata-data-distribution.png" />
</Frame>

The panel opens in **Metadata distribution** mode and shows a bar chart of how your samples are distributed across the values of a chosen metadata key (for example: `accent`, `site`, `subject`). Use the controls at the top right to:

* **All samples / Training / Test**: scope the view to your full dataset or just one split.
* **Overall / Compare**: switch between the single-key bar chart and a cross-tabulation of two metadata keys (see below).
* **Count / %**: toggle between absolute sample counts and relative percentages.
* **Metadata key selector**: pick which metadata key(s) to visualise.

#### Comparing two metadata keys

Switch to **Compare** mode to cross-tabulate two metadata keys at once. The result is a heatmap table where rows represent values of the first key and columns represent values of the second key, with cell shading proportional to the amount of data.

<Frame caption="Data distribution panel: Compare mode cross-tabulating two metadata keys">
  <img src="https://mintcdn.com/edgeimpulse/7F52kDOIkzrKqXkS/.assets/images/metadata-data-distribution-compare.png?fit=max&auto=format&n=7F52kDOIkzrKqXkS&q=85&s=c5dc5d51aa928b0391e0a271bf9d4f61" alt="Compare mode chart cross-tabulating two metadata keys in the data distribution panel" width="1889" height="1013" data-path=".assets/images/metadata-data-distribution-compare.png" />
</Frame>

A **Coverage** summary is shown above the table (for example: `Coverage: 92.5% (32m 57s / 35m 37s) • accent: 12 unique • gender: 2 unique`) telling you what proportion of your dataset has both of the selected metadata keys defined and how many unique values each key has.

### Adding metadata when adding data

It's pretty unpractical to manually add metadata to each data item, so the easiest way is to add metadata when you upload data. You can do this either by:

1. Providing an [info file](/tools/clis/edge-impulse-cli/uploader#custom-labeling-and-metadata) file when uploading data (this works both in the CLI and in the Studio).
2. Setting the `x-metadata` header to a JSON string when calling the ingestion service:

```
curl -X POST \
     -H "x-api-key: ei_238fae..." \
     -H "x-label: car" \
     -H "x-metadata: '{\"site\":\"Paris\"}' \
     -H "Content-Type: multipart/form-data" \
     -F "data=@one.png" \
     https://ingestion.edgeimpulse.com/api/training/files
```

### Reading and writing metadata through the API

You can read samples, including their metadata via the [List samples](/apis/studio/raw-data/list-samples) API call, and then use the [Set sample metadata](/apis/studio/raw-data/set-sample-metadata) API to update the metadata. For example, this is how you add a metadata field to the first data sample in your project using the [Python API Bindings](/tutorials/tools/api-bindings/studio/python/use-python-api-bindings):

```python theme={"system"}
import edgeimpulse_api as ei

# update project ID / API Key
EI_PROJECT_ID = 1
EI_API_KEY = "ei_8b8..."

# instantiate the API client
configuration = ei.Configuration()
configuration.api_key["ApiKeyAuthentication"] = EI_API_KEY

api = ei.ApiClient(configuration)
raw_data = ei.RawDataApi(api)

# fetch the first page of data
samples = raw_data.list_samples(project_id=EI_PROJECT_ID, category='training', offset=0, limit=20)

# grab the current metadata
metadata = samples.samples[0].metadata
print('first sample metadata is', metadata)

# add an extra key
metadata = metadata if metadata else {}
metadata['hello'] = 'world'

# update metadata
raw_data.set_sample_metadata(project_id=EI_PROJECT_ID,
                             sample_id=samples.samples[0].id,
                             set_sample_metadata_request=ei.SetSampleMetadataRequest(metadata=metadata))
print('updated metadata!')

```

### Using metadata to control your train validation split

When training an ML model we split your data into a train and a validation set. This is done so that during training you can evaluate whether your model works on data that it has seen before (train set) and on data that it has never seen before (validation set) - ideally your model performs similarly well on both data sets: a sign that your model will perform well in the field on completely novel data.

However, this can give a false sense of security if data that is very similar ends up in both your train and validation set ("data leakage"). For example:

* You split a video into individual frames. These images don't differ much from frame to frame; and you don't want some frames in the train, and some in the validation set.
* You're building a sleep staging algorithm, and look at 30 second windows. From window to window the data for one person will look similar, so you don't want one window in the train, another in the validation set for the same person in the same night.

By default we split your training data randomly in a train and validation set (80/20 split) - which does not prevent data leakage, but if you tag your data items with metadata you can avoid this. To do so:

1. Tag all your data items with metadata.
2. Go to any ML block and under *Advanced training settings* set 'Split train/validation set on metadata key' to a metadata key (f.e. `video_file`).

   <Frame caption="Controlling the train/validation split with a metadata key">
     <img src="https://mintcdn.com/edgeimpulse/ydYuX7QIsmo2tzb8/.assets/images/using-metadata-to-control-train-val-split.png?fit=max&auto=format&n=ydYuX7QIsmo2tzb8&q=85&s=50ca3fbae0decb06998baa98b35f6898" alt="Dataset split dialog using a metadata key to control train and validation grouping" width="1260" height="488" data-path=".assets/images/using-metadata-to-control-train-val-split.png" />
   </Frame>

Now every data item with the same *metadata value* for `video_file` will always be grouped together in either the train or the validation set; so no more data leakage.
