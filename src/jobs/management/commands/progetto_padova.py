import lime
import lime.lime_tabular
import matplotlib as plt
from django.core.management.base import BaseCommand
from sklearn.externals import joblib

from scripts.progetto_padova import progetto_padova
from src.core.core import get_encoded_logs
from src.jobs.models import Job


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        progetto_padova()
