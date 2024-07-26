from django.shortcuts import render, get_object_or_404, redirect
from .models import Product, ReviewRating
from category.models import Category
from carts.models import CartItem
from carts.views import _cart_id
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.db.models import Q
from .forms import ReviewForm
from django.contrib import messages
from orders.models import OrderProduct
from tqdm import tqdm
import os
import bz2
import re
import tensorflow as tf
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dropout,Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
from keras import backend as K


def store(request, category_slug = None):
    categories = None
    products = None

    if category_slug != None:
        categories = get_object_or_404(Category, slug = category_slug)
        products = Product.objects.filter(category = categories, is_available = True)
        paginator = Paginator(products, 6)
        page = request.GET.get('page')
        paged_products = paginator.get_page(page)
        product_count = products.count()
    else:
        products = Product.objects.all().filter(is_available = True).order_by('id')
        paginator = Paginator(products, 6)
        page = request.GET.get('page')
        paged_products = paginator.get_page(page)
        product_count = products.count()

    context = {
    'products': paged_products,
    'product_count' : product_count,
    }

    return render(request, 'store/store.html', context)

def product_detail(request, category_slug, product_slug):
    try:
        single_product = Product.objects.get(category__slug = category_slug, slug = product_slug)
        in_cart = CartItem.objects.filter(cart__cart_id = _cart_id(request), product = single_product).exists()
    except Exception as e:
        raise e

    if request.user.is_authenticated:
        try:
            orderproduct = OrderProduct.objects.filter(user = request.user, product_id = single_product.id).exists()
        except OrderProduct.DoesNotExist:
            orderproduct = None
    else:
        orderproduct = None

    reviews = ReviewRating.objects.filter(product_id = single_product.id, status = True)


    context = {
    'single_product' : single_product,
    'in_cart' : in_cart,
    'orderproduct' : orderproduct,
    'reviews' : reviews,
    }

    return render(request, 'store/product_detail.html', context)

def search(request):
    if 'keyword' in request.GET:
        keyword = request.GET['keyword']
        if keyword:
            products = Product.objects.order_by('created_date').filter(Q(description__icontains = keyword) | Q(product_name__icontains = keyword))
            product_count = products.count()
    context = {
    'products' : products,
    'product_count' : product_count,
    }
    return render(request, 'store/store.html', context)

def submit_review(request, product_id):

    model = load_model('store/LSTMmodel.h5')
    train_file = bz2.BZ2File('store/test.ft.txt.bz2')

    train_file_lines = train_file.readlines()
    train_file_lines = [x.decode('utf-8') for x in train_file_lines]
    train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
    train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]

    for i in range(len(train_sentences)):
        train_sentences[i] = re.sub('\d', '0', train_sentences[i])

    for i in range(len(train_sentences)):
        if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
            train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])
    X_train, X_test, y_train, y_test = train_test_split(train_sentences, train_labels, train_size=0.80, test_size=0.20, random_state=42)
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)

    def rate(p):
        return (p*5)

    url = request.META.get('HTTP_REFERER')
    if request.method == 'POST':
        try:
            reviews = ReviewRating.objects.get(user__id = request.user.id, product__id = product_id)
            form = ReviewForm(request.POST, instance = reviews)

            if form.is_valid():
                comment = form.cleaned_data['subject']

            review = [comment]
            print("Comment:", comment)

            pred = model.predict(pad_sequences(tokenizer.texts_to_sequences(review), maxlen=100))
            print("Prediction:", pred)

            rating = rate(pred.item(0, 0))
            print('Rating:', rating)

            if rating > 4.5 and rating <= 5.0:
                rating = 5.0
            elif rating > 4.0 and rating <= 4.5:
                rating = 4.5
            elif rating > 3.5 and rating <= 4.0:
                rating = 4.0
            elif rating > 3.0 and rating <= 3.5:
                rating = 3.5
            elif rating > 2.5 and rating <= 3.0:
                rating = 3.0
            elif rating > 2.0 and rating <= 2.5:
                rating = 2.5
            elif rating > 1.5 and rating <= 2.0:
                rating = 2.0
            elif rating > 1.0 and rating <= 1.5:
                rating = 1.5
            elif rating > 0.5 and rating <= 1.0:
                rating = 1.0
            elif rating > 0.0 and rating <= 0.5:
                rating = 0.5
            elif rating < 0.5:
                rating = 0.0
            print('Rating:', rating)

            reviews.rating = rating
            form.save()
            return redirect(url)

        except ReviewRating.DoesNotExist:
            form = ReviewForm(request.POST)
            if form.is_valid():
                data = ReviewRating()

                data.subject = form.cleaned_data['subject']
                data.review = form.cleaned_data['review']

                comment = form.cleaned_data['subject']
                print("Comment:", comment)
                review = [comment]

                pred = model.predict(pad_sequences(tokenizer.texts_to_sequences(review), maxlen=100))
                rating = rate(pred.item(0, 0))
                print('Rating:', rating)

                if rating > 4.5 and rating <= 5.0:
                    rating = 5.0
                elif rating > 4.0 and rating <= 4.5:
                    rating = 4.5
                elif rating > 3.5 and rating <= 4.0:
                    rating = 4.0
                elif rating > 3.0 and rating <= 3.5:
                    rating = 3.5
                elif rating > 2.5 and rating <= 3.0:
                    rating = 3.0
                elif rating > 2.0 and rating <= 2.5:
                    rating = 2.5
                elif rating > 1.5 and rating <= 2.0:
                    rating = 2.0
                elif rating > 1.0 and rating <= 1.5:
                    rating = 1.5
                elif rating > 0.5 and rating <= 1.0:
                    rating = 1.0
                elif rating > 0.0 and rating <= 0.5:
                    rating = 0.5
                elif rating < 0.5:
                    rating = 0.0

                print('Rating:', rating)

                data.rating = rating

                data.ip = request.META.get('REMOTE_ADDR')
                data.product_id = product_id
                data.user_id = request.user.id
                data.save()
                messages.success(request, 'Thank You! Review Recorded')
                return redirect(url)
