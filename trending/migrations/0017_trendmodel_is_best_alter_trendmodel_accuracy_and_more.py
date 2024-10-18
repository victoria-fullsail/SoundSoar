# Generated by Django 4.2.9 on 2024-10-06 21:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trending', '0016_remove_playlist_description_customplaylist'),
    ]

    operations = [
        migrations.AddField(
            model_name='trendmodel',
            name='is_best',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='trendmodel',
            name='accuracy',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='trendmodel',
            name='f1_score',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='trendmodel',
            name='model_type',
            field=models.CharField(choices=[('RandomForest', 'RandomForest'), ('HistGradientBoost', 'HistGradientBoost'), ('LogisticRegression', 'LogisticRegression'), ('SVM', 'SVM'), ('KNN', 'KNN')], max_length=25),
        ),
        migrations.AlterField(
            model_name='trendmodel',
            name='precision',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='trendmodel',
            name='recall',
            field=models.FloatField(blank=True, null=True),
        ),
    ]