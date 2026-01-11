"""Rate limiting configuration."""
from slowapi import Limiter
from slowapi.util import get_remote_address

# Create limiter instance that will be shared across the app
limiter = Limiter(key_func=get_remote_address)
