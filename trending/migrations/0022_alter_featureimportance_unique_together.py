# Generated by Django 4.2.9 on 2024-10-16 23:32

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('trending', '0021_trendmodel_best_paramters_and_more'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='featureimportance',
            unique_together=set(),
        ),
    ]
