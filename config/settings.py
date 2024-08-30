import os
from pathlib import Path
from decouple import config


BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = config('SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# SERVER SETTINGS
ALLOWED_HOSTS = ['localhost', '3.130.223.246', 'soundsoar.com', 'www.soundsoar.com']
CSRF_TRUSTED_ORIGINS = ['https://localhost', 'https://3.130.223.246','https://soundsoar.com', 'https://www.soundsoar.com']
SECURE_CROSS_ORIGIN_OPENER_POLICY = None
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')


# Application definition

INSTALLED_APPS = [
    'jazzmin',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
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
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

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
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

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


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# STATIC FILES (CSS, JavaScript, images)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles') # Directory where collectstatic will collect static files for production


# MEDIA FILES (user uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media') # Directory where uploaded files will be stored

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

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
        {"app": "auth"},  # Keep only necessary apps
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
