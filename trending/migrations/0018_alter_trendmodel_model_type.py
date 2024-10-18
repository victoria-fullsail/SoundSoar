# Generated by Django 4.2.9 on 2024-10-06 23:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trending', '0017_trendmodel_is_best_alter_trendmodel_accuracy_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trendmodel',
            name='model_type',
            field=models.CharField(choices=[('RandomForest', 'RandomForest'), ('HistGradientBoost', 'HistGradientBoost'), ('LogisticRegression', 'LogisticRegression'), ('SVM', 'SVM'), ('LightGBM', 'LightGBM')], max_length=25),
        ),
    ]