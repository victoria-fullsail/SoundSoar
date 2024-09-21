# Generated by Django 4.2.9 on 2024-09-21 19:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trending', '0013_trackfeatures_predicted_trend_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trackfeatures',
            name='predicted_trend',
            field=models.CharField(blank=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')], max_length=10, null=True),
        ),
        migrations.AlterField(
            model_name='trackfeatures',
            name='trend',
            field=models.CharField(blank=True, choices=[('up', 'Up'), ('down', 'Down'), ('stable', 'Stable')], max_length=10, null=True),
        ),
    ]
