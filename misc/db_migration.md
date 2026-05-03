I have a project on Google Cloud called "deminimishelper". I want to transfter the project owned by me under my school gmail account to my person gmail account. I want to do it on CLI. Return detailed instructions. First, check if I have priviledges to transfer projects across organizations through CLI.

```bash
# Check active account and project
gcloud auth list
gcloud config get-value project
gcloud config get-value account

# Set the project if needed
gcloud config set project deminimishelper

# Check your IAM roles on the project
gcloud projects get-iam-policy deminimishelper --flatten="bindings[].members" --format="table(bindings.role, bindings.members)" --filter="bindings.members:ximeng@ucdavis.edu"

# Check if you have the specific permission needed for project migration
gcloud projects get-iam-policy deminimishelper \
  --format=json | grep -E "(owner|roles/resourcemanager)"

# Check org-level permissions (needed if project is under an org)
gcloud organizations list

# If you get an org ID, check your org-level role
gcloud organizations get-iam-policy 558550560619 --flatten="bindings[].members" --filter="bindings.members:ximeng@ucdavis.edu" --format="table(bindings.role, bindings.members)"

```

# migration plan

```bash
gcloud auth login xiangtaom@gmail.com
gcloud config set account xiangtaom@gmail.com


gcloud projects create deminimishelper-v2
gcloud config set project deminimishelper-v2

gcloud storage buckets list --project=deminimishelper-v2
gcloud datastore indexes list --project=deminimishelper-v2


# Link billing
gcloud billing accounts list
gcloud billing projects link deminimishelper-v2 --billing-account=01CC3F-FC9755-C21974


# Grant school account write access to destination bucket
gcloud storage buckets add-iam-policy-binding gs://deminimishelper-v2 --member="user:ximeng@ucdavis.edu" --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding deminimishelper-v2 --member="user:ximeng@ucdavis.edu" --role="roles/serviceusage.serviceUsageConsumer"

# Enable required APIs
gcloud services enable bigquery.googleapis.com datastore.googleapis.com storage.googleapis.com dataform.googleapis.com
```
```bash
# Switch back to school account to read source
gcloud config set account ximeng@ucdavis.edu

# List buckets
gcloud storage buckets list --project=deminimishelper

# Copy each bucket to new project (replace BUCKET_NAME)
gcloud storage cp -r gs://deminimishelper gs://deminimishelper-v2

# Or transfer entire bucket using storage transfer
gcloud storage buckets create gs://deminimishelper-v2 --project=deminimishelper-v2
gsutil -m cp -r gs://deminimishelper/* gs://deminimishelper-v2/


gcloud transfer jobs create gs://deminimishelper/models/ gs://deminimishelper-v2/models/ --source-creds-file=none

# use the -j (parallelism) flag with gcloud storage cp:
gcloud storage cp -r gs://deminimishelper/models/ gs://deminimishelper-v2/models/ -j 32

```