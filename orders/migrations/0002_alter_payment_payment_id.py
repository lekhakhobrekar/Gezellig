# Generated by Django 4.0.4 on 2022-06-19 09:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('orders', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='payment',
            name='payment_id',
            field=models.CharField(blank=True, max_length=100),
        ),
    ]
