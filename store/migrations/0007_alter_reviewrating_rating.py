# Generated by Django 4.0.4 on 2022-07-09 10:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('store', '0006_alter_reviewrating_rating'),
    ]

    operations = [
        migrations.AlterField(
            model_name='reviewrating',
            name='rating',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
