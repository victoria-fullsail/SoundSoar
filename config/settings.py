import os
from pathlib import Path
from decouple import config

# BASIC SETTINGS
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = config('SECRET_KEY')
DEBUG = True
SITE_ID = 1

# SERVER SETTINGS
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '3.130.223.246', 'soundsoar.com', 'www.soundsoar.com']
CSRF_TRUSTED_ORIGINS = ['https://localhost', 'https://3.130.223.246','https://soundsoar.com', 'https://www.soundsoar.com']
SECURE_CROSS_ORIGIN_OPENER_POLICY = None
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

CORS_ALLOW_ALL_ORIGINS = True


INTERNAL_IPS = [
    '127.0.0.1',
]


# Application definition
INSTALLED_APPS = [
    'jazzmin',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sites',
    'debug_toolbar',
    'corsheaders',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.spotify',
    'core.apps.CoreConfig',
    'trending.apps.TrendingConfig',
    'popularity.apps.PopularityConfig',
    'personalized.apps.PersonalizedConfig', 
    'userpref.apps.UserprefConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'debug_toolbar.middleware.DebugToolbarMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'allauth.account.middleware.AccountMiddleware',
    'corsheaders.middleware.CorsMiddleware',
]

AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
)


ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'


# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': config('MYSQL_NAME'),
        'USER': config('MYSQL_USER'),
        'PASSWORD': config('MYSQL_PASSWORD'),
        'HOST': config('MYSQL_HOST'),
        'PORT': config('MYSQL_PORT'),
    }
}


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Allauth Settings
LOGIN_URL = 'accounts/login/'
LOGIN_REDIRECT_URL = 'core:home'
ACCOUNT_EMAIL_REQUIRED = True
ACCOUNT_USERNAME_REQUIRED = False
ACCOUNT_AUTHENTICATION_METHOD = 'email'
SOCIALACCOUNT_ENABLED = True
SOCIALACCOUNT_AUTO_SIGNUP = True
ACCOUNT_EMAIL_SUBJECT_PREFIX = "[SoundSoar] "
ACCOUNT_UNIQUE_EMAIL = True


# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Pacific/Easter'
USE_I18N = True
USE_TZ = True


# STATIC FILES (CSS, JavaScript, images)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles') # Directory where collectstatic will collect static files for production

# AWS S3 settings for public media files
AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
AWS_S3_REGION_NAME = config('AWS_S3_REGION_NAME')
AWS_DEFAULT_ACL = 'public-read'
AWS_S3_OBJECT_PARAMETERS = {
    'CacheControl': 'max-age=86400',
}

# Media files (uploads by users)
AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
AWS_S3_CUSTOM_DOMAIN = f"{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com"
DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
MEDIA_URL = f"https://{AWS_S3_CUSTOM_DOMAIN}/"
MEDIA_ROOT = ''  # This can be left empty

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# SPOTIPY
SPOTIPY_CLIENT_ID = config('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = config('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = config('SPOTIPY_REDIRECT_URI')

# Spotify SSO settings
SOCIALACCOUNT_PROVIDERS = {
    'spotify': {
        'SCOPE': ['user-read-email', 'playlist-read-private', 'playlist-modify-private', 'user-top-read'],
        'AUTH_PARAMS': {'response_type': 'code'},
        'APP': {
            'client_id': config('SPOTIPY_CLIENT_ID'),
            'secret': config('SPOTIPY_CLIENT_SECRET'),
            'key': '',
        },
        'VERIFIED_EMAIL': True,
    }
}



SPOTIFY_LOG_FILE_PATH = os.path.join(BASE_DIR, 'logs', 'spotify.log')

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',  # Changed to DEBUG to see all messages in console
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': SPOTIFY_LOG_FILE_PATH,
            'maxBytes': 5 * 1024 * 1024,  # 5 MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'spotify': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

# Jazzmin Settings
JAZZMIN_SETTINGS = {
    'site_title': 'SoundSoar',
    'site_header': 'SoundSoar',
    'site_logo': 'core/img/logo2.png',
    "login_logo": 'core/img/logo2.png',
    "welcome_sign": "Welcome, please login with your SoundSoar credentials!",
    "search_model": [],  # Keep empty as it's for superusers only
    "changeform_format": "horizontal_tabs",

    ############
    # Top Menu #
    ############

    "topmenu_links": [
    {"name": "Home", "url": "admin:index"},
    {"app": "sites"},  # Add the sites app to the top menu
    ],

    #############
    # Side Menu #
    #############

    "show_sidebar": True,
    "navigation_expanded": False,
    "hide_apps": [],

    # Additional settings
    "show_ui_builder": False,
}

JAZZMIN_UI_TWEAKS = {
    "navbar_small_text": False,
    "footer_small_text": True,
    "body_small_text": False,
    "brand_small_text": False,
    "brand_colour": "navbar-black",  # Black brand color to match background
    "accent": "#0077b6",  # Blue accent for SoundSoar
    "navbar": "navbar-black navbar-dark",  # Black navbar with white text
    "no_navbar_border": True,
    "navbar_fixed": True,
    "layout_boxed": False,
    "footer_fixed": False,
    "sidebar_fixed": True,
    "sidebar": "sidebar-dark-primary",  # Black sidebar to match theme
    "sidebar_nav_small_text": False,
    "sidebar_disable_expand": False,
    "sidebar_nav_child_indent": True,
    "sidebar_nav_compact_style": False,
    "sidebar_nav_legacy_style": False,
    "sidebar_nav_flat_style": False,
    "theme": "default",
    "dark_mode_theme": None,
    "button_classes": {
        "primary": "btn-primary",
        "secondary": "btn-outline-secondary",
        "info": "btn-info",
        "warning": "btn-warning",
        "danger": "btn-danger",
        "success": "btn-success"
    },
    "text_color": {
        "navbar": "text-light",
        "footer": "text-light",
        "body": "text-white",  # White text for readability on dark background
        "brand": "text-light"
    },
    "background_color": {
        "body": "bg-black",  # Set the main background to black
        "navbar": "bg-black",  # Navbar background to black
        "sidebar": "bg-black",  # Sidebar background to black
    },
    "button_color": {
        "primary": "btn-primary",
        "secondary": "btn-outline-secondary",
        "info": "btn-info",
        "warning": "btn-warning",
        "danger": "btn-danger",
        "success": "btn-success",
        "accent": "#0077b6",  # Accent color for SoundSoar buttons
    }
}
