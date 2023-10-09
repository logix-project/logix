import hashlib


class HasherBase:
    def hash(self, data):
        """Hash the provided data."""
        raise NotImplementedError


class SHA256Hasher(HasherBase):
    def hash(self, data):
        """Hash using SHA-256."""
        return hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()
