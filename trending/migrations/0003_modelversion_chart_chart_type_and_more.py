# Generated by Django 5.1 on 2024-09-01 17:46

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trending', '0002_chart_track_remove_sound_platform_playlist_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelVersion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('version_number', models.CharField(max_length=50, unique=True)),
                ('description', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('is_active', models.BooleanField(default=False)),
            ],
        ),
        migrations.AddField(
            model_name='chart',
            name='chart_type',
            field=models.CharField(choices=[('custom', 'Custom'), ('spotify_playlist', 'Spotify Playlist')], default='spotify_playlist', max_length=20),
        ),
        migrations.AddField(
            model_name='track',
            name='added_to_playlists_count',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='track',
            name='stream_change',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='track',
            name='streams',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='track',
            name='updated_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='chart',
            name='created_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='playlist',
            name='chart',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='playlists', to='trending.chart'),
        ),
        migrations.AlterField(
            model_name='playlist',
            name='created_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.CreateModel(
            name='ModelPerformance',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('accuracy', models.FloatField()),
                ('precision', models.FloatField()),
                ('recall', models.FloatField()),
                ('f1_score', models.FloatField()),
                ('evaluation_date', models.DateTimeField(default=django.utils.timezone.now)),
                ('model_version', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='performances', to='trending.modelversion')),
            ],
        ),
        migrations.CreateModel(
            name='PlaylistTrack',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('added_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('playlist', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='playlist_tracks', to='trending.playlist')),
                ('track', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='playlist_tracks', to='trending.track')),
            ],
        ),
        migrations.CreateModel(
            name='StreamHistory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(default=django.utils.timezone.now)),
                ('streams', models.IntegerField()),
                ('track', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='stream_history', to='trending.track')),
            ],
            options={
                'ordering': ['-timestamp'],
            },
        ),
        migrations.CreateModel(
            name='TrackFeatures',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('danceability', models.FloatField(blank=True, null=True)),
                ('energy', models.FloatField(blank=True, null=True)),
                ('tempo', models.FloatField(blank=True, null=True)),
                ('current_streams', models.IntegerField()),
                ('streams_last_24h', models.IntegerField()),
                ('streams_last_7d', models.IntegerField()),
                ('streams_last_30d', models.IntegerField()),
                ('current_popularity', models.IntegerField()),
                ('velocity', models.FloatField()),
                ('trend', models.CharField(choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')], max_length=50)),
                ('updated_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('track', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='features', to='trending.track')),
            ],
        ),
    ]
