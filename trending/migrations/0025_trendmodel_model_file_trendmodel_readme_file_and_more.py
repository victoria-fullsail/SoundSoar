# Generated by Django 4.2.9 on 2024-10-17 21:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trending', '0024_trendmodel_confusion_matrix_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='trendmodel',
            name='model_file',
            field=models.FileField(blank=True, null=True, upload_to='trending/trend_model/models/'),
        ),
        migrations.AddField(
            model_name='trendmodel',
            name='readme_file',
            field=models.FileField(blank=True, null=True, upload_to='trending/trend_model/models/'),
        ),
        migrations.AlterField(
            model_name='trendmodel',
            name='csv_data',
            field=models.FileField(blank=True, null=True, upload_to='trending/trend_model/csv_data/'),
        ),
    ]
