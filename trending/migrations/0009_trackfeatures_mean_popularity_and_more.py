# Generated by Django 4.2.9 on 2024-09-12 04:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trending', '0008_remove_track_added_to_playlists_count_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='trackfeatures',
            name='mean_popularity',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trackfeatures',
            name='median_popularity',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trackfeatures',
            name='speechiness',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trackfeatures',
            name='std_popularity',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trackfeatures',
            name='valence',
            field=models.FloatField(blank=True, null=True),
        ),
    ]