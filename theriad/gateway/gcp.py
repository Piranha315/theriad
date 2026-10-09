"""Optional Google implementations. Imports do not require cloud dependencies."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone


class GoogleCredentials:
    def __init__(self, credentials=None, audience=None):
        self.credentials = credentials
        self.audience = audience
        self.lock = threading.Lock()

    def token(self):
        import google.auth
        from google.auth.transport.requests import Request

        with self.lock:
            if self.credentials is None:
                if self.audience:
                    from google.auth import impersonated_credentials

                    source, _ = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                    import os

                    account = os.environ.get("THERIAD_BRIDGE_SERVICE_ACCOUNT")
                    if not account:
                        raise ValueError("bridge service account is required for keyless ID tokens")
                    impersonated = impersonated_credentials.Credentials(
                        source,
                        account,
                        target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    self.credentials = impersonated_credentials.IDTokenCredentials(
                        impersonated, target_audience=self.audience, include_email=True
                    )
                else:
                    self.credentials, _ = google.auth.default(
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
            if not self.credentials.valid or self.credentials.expired:
                self.credentials.refresh(Request())
            return self.credentials.token


class GoogleStore:
    """Firestore transactions for coordination, GCS for potentially large records."""

    def __init__(self, project, bucket):
        from google.cloud import firestore, storage

        self.firestore = firestore.Client(project=project)
        self.bucket = storage.Client(project=project).bucket(bucket)
        self.collection = self.firestore.collection("theriad_gateway")

    def reference(self, key):
        return self.collection.document(hashlib.sha256(key.encode()).hexdigest())

    def get(self, key):
        document = self.reference(key).get()
        if not document.exists:
            return None
        record = document.to_dict()
        if record.get("expires", float("inf")) <= time.time():
            return None
        if "blob" in record:
            return json.loads(self.bucket.blob(record["blob"]).download_as_bytes())
        return record["value"]

    def put(self, key, value, ttl=None):
        data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        blob_name = (
            "ephemeral/" + uuid.uuid4().hex
            if ttl
            else "records/" + hashlib.sha256(data).hexdigest()
        )
        blob = self.bucket.blob(blob_name)
        from google.api_core.exceptions import PreconditionFailed

        try:
            blob.upload_from_string(data, content_type="application/json", if_generation_match=0)
        except PreconditionFailed:
            pass
        record = {"blob": blob_name}
        if ttl:
            record["expires"] = time.time() + ttl
            record["expires_at"] = datetime.fromtimestamp(record["expires"], timezone.utc)
        self.reference(key).set(record)

    def mutate(self, key, fn):
        from google.cloud import firestore

        reference = self.reference(key)

        @firestore.transactional
        def transaction_change(transaction):
            doc = reference.get(transaction=transaction)
            old = doc.to_dict() if doc.exists else None
            if old and old.get("expires", float("inf")) <= time.time():
                old = None
            if old and "blob" in old:
                current = json.loads(self.bucket.blob(old["blob"]).download_as_bytes())
            else:
                current = old["value"] if old else None
            result = fn(current)
            # Coordination state is small; raw messages are stored only in GCS.
            record = {"value": result}
            if old and "expires" in old:
                record.update(expires=old["expires"], expires_at=old["expires_at"])
            transaction.set(reference, record)
            return result

        return transaction_change(self.firestore.transaction())
