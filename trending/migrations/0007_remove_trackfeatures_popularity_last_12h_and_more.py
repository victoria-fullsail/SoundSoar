# Generated by Django 4.2.9 on 2024-09-12 03:20

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('trending', '0006_playlist_tracks_delete_playlisttrack'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='trackfeatures',
            name='popularity_last_12h',
        ),
        migrations.RemoveField(
            model_name='trackfeatures',
            name='popularity_last_24h',
        ),
        migrations.RemoveField(
            model_name='trackfeatures',
            name='popularity_last_3d',
        ),
        migrations.RemoveField(
            model_name='trackfeatures',
            name='popularity_last_3h',
        ),
        migrations.RemoveField(
            model_name='trackfeatures',
            name='popularity_last_5d',
        ),
        migrations.RemoveField(
            model_name='trackfeatures',
            name='popularity_last_6h',
        ),
    ]
