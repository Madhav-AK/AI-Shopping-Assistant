# External Packages:
from __future__ import annotations
import gradio as gr
import os
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Tuple, Union
from PIL import Image

# Importing from our various other files:
import my_clipmlp
import yolo_shoe_detection
import llm
from llm import beginning_llm_prompt, second_llm_prompt, clip_category_prompt, clip_preferences_prompt
myLLM = llm.MyLLMClass()


# Find the root of the project folder
project_root = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(project_root, "fakeStore", "static", "uploads", "metadata.json")
with open(metadata_path, "r") as f:
    products = json.load(f)

# dict for fast lookup by id
product_by_id = {prod["id"]: prod for prod in products}

# dict for fast lookup by filename
product_by_filename = {prod["filename"]: prod for prod in products}

def get_price(product_id):
    product = product_by_id.get(product_id)
    if product:
        return product["price_cents"] / 100  # convert to dollars if needed
    return None

def get_id_by_filename(filename):
    product = product_by_filename.get(filename)
    return product["id"]

    

LocalImg = Union[str, Image.Image]

@dataclass
class Product:
    image: LocalImg
    price: str
    name: str = ""
    link: str = ""

N = 24

# -----------------------------------------------------------------------------
# Core Backend Calling
# -----------------------------------------------------------------------------

def generate_initial_recommendations(img_path: str | None, n = 8) -> Tuple[List[Product], str]:
    image = Image.open(img_path).copy()
    cropped_images = yolo_shoe_detection.get_yolo_cropped_images(image)

    # Instance-Aware Retrieval Distributor
    topn = []
    already_suggested_names = set()
    other_possibilities = []

    base, rem = divmod(n, len(cropped_images)) 

    for i, img in enumerate(cropped_images):
        k = base + (1 if i < rem else 0)
        clipmlp_category, clip_feat = my_clipmlp.classify_image_clipmlp(img)

        j = 0
        for possibility in my_clipmlp.clip_find_top_k_similar_in_category(clipmlp_category, clip_feat, None, None, N):
            if j < k:
                name, sim, feat = possibility
                if name not in already_suggested_names:
                    already_suggested_names.add(name)

                    # Merge: build Product here
                    product_file_name = os.path.basename(name)[:-4] + '.jpg'
                    product_image_path = f"shoes\\{clipmlp_category}\\{product_file_name}"
                    product_image = Image.open(product_image_path)
                    product_price = f"₹{my_clipmlp.brand_mapping[clipmlp_category.replace('\\','_')][product_file_name][1]}"
                    product_id  = get_id_by_filename(product_file_name)
                    product_link = f"http://127.0.0.1:5000/item/{product_id}"
                    product = Product(product_image, product_price, link=product_link)
                    topn.append((product, sim, feat))  # Save product instead of filename
            else:
                other_possibilities.append(possibility)
            j += 1
            if j == n:
                break

        for possibility in other_possibilities:
            name, sim, feat = possibility
            if name not in already_suggested_names:
                already_suggested_names.add(name)

                # Merge: build Product here
                product_file_name = os.path.basename(name)[:-4] + '.jpg'
                product_image_path = f"shoes\\{clipmlp_category}\\{product_file_name}"
                product_image = Image.open(product_image_path)
                product_price = f"₹{my_clipmlp.brand_mapping[clipmlp_category.replace('\\','_')][product_file_name][1]}"
                product_id  = get_id_by_filename(product_file_name)
                product_link = f"http://127.0.0.1:5000/item/{product_id}"
                product = Product(product_image, product_price, link=product_link)

                topn.append((product, sim, feat))
            j += 1
            if j == n:
                break

    # Contrastive Re-Ranking using LLM
    myLLM.create_new_chat()
    llm_response = myLLM.query_chat(beginning_llm_prompt, image)
    topn = my_clipmlp.contrastive_reranking(topn, llm_response)

    prods = [tup[0] for tup in topn]  # Now we only extract the Product from each tuple

    # LLM context update
    myLLM.clipcategory, myLLM.last_clip_feat = my_clipmlp.classify_image_clipmlp(image)
    myLLM.last_pricerange = None
    myLLM.last_brand = None
    myLLM.retrieved_feats = [tup[2] for tup in topn]

    return prods, llm_response

def refine_recommendations(msg: str, hist: List[Tuple[str, str]], prods: List[Product]) -> Tuple[List[Product], str]:
    if myLLM.chat_msg_count == 1:
        llm_response = myLLM.query_chat(second_llm_prompt + msg + clip_category_prompt + myLLM.clipcategory.replace('\\','_')
        + clip_preferences_prompt + myLLM.preferences)
    else:
        llm_response = myLLM.query_chat(msg + '\n' + clip_category_prompt + myLLM.clipcategory.replace('\\','_') 
        + clip_preferences_prompt + myLLM.preferences)

    category, query, pricerange, brand, preferences, msg = myLLM.extract_data_from_followup_responses(llm_response)
    # print('-------Data retrieved from user msg-------')
    # print(category)
    # print(query)
    # print(pricerange)
    # print(brand)
    # print(preferences)
    # print('--------------')
    feat = my_clipmlp.encode_one_text(query)
    topn = my_clipmlp.clip_find_top_k_similar_in_category(category, feat, pricerange, brand, N)
    myLLM.clipcategory = category
    myLLM.last_clip_feat = feat
    myLLM.last_pricerange = pricerange
    myLLM.last_brand = brand
    myLLM.preferences = preferences
    myLLM.retrieved_feats = [tup[2] for tup in topn]

    prods = []
    for rank, (p, sim, feat) in enumerate(topn, start=1):
        product_file_name = os.path.basename(p)[:-4] + '.jpg'
        product_price = f"₹{my_clipmlp.brand_mapping[category.replace('\\','_')][product_file_name][1]}"
        product_id  = get_id_by_filename(product_file_name)
        product_link = f"http://127.0.0.1:5000/item/{product_id}"
        prods.append(Product(Image.open(f"shoes\\{category}\\{product_file_name}"), product_price, link=product_link))

    if prods == []:
        msg = "I'm sorry, we don't have any available stock of that particular brand at your price range. Please try something else."
    
    return prods, msg

def recommend_less(idx, prods):
    prod_selected_name = prods[idx].name
    myLLM.last_clip_feat = 0.7*np.asarray(myLLM.last_clip_feat) + 0.3*np.asarray(myLLM.retrieved_feats[idx])
    category = myLLM.clipcategory
    feat = myLLM.last_clip_feat
    pricerange = myLLM.last_pricerange
    brand = myLLM.last_brand 
    topn = my_clipmlp.clip_find_top_k_similar_in_category(category, feat, pricerange, brand, k=N)
    prods = []
    for rank, (p, sim, feat) in enumerate(topn, start=1):
        product_file_name = os.path.basename(p)[:-4] + '.jpg'
        product_price = f"₹{my_clipmlp.brand_mapping[category.replace('\\','_')][product_file_name][1]}"
        product_id  = get_id_by_filename(product_file_name)
        product_link = f"http://127.0.0.1:5000/item/{product_id}"
        prods.append(Product(Image.open(f"shoes\\{category}\\{product_file_name}"), product_price, link=product_link))

    return prods, "Got it! We'll tune our recommendations accordingly!"

def recommend_more(idx, prods):
    prod_selected_name = prods[idx].name
    myLLM.last_clip_feat = 1.3*np.asarray(myLLM.last_clip_feat) - 0.3*np.asarray(myLLM.retrieved_feats[idx])
    category = myLLM.clipcategory
    feat = myLLM.last_clip_feat
    pricerange = myLLM.last_pricerange
    brand = myLLM.last_brand 
    topn = my_clipmlp.clip_find_top_k_similar_in_category(category, feat, pricerange, brand, k=N)
    prods = []
    for rank, (p, sim, feat) in enumerate(topn, start=1):
        product_file_name = os.path.basename(p)[:-4] + '.jpg'
        product_price = f"₹{my_clipmlp.brand_mapping[category.replace('\\','_')][product_file_name][1]}"
        product_id  = get_id_by_filename(product_file_name)
        product_link = f"http://127.0.0.1:5000/item/{product_id}"
        prods.append(Product(Image.open(f"shoes\\{category}\\{product_file_name}"), product_price, link=product_link))

    return prods, "Got it! We'll tune our recommendations accordingly!"

# ----------------------------------------------------------------------------- #
#  UI 
# ----------------------------------------------------------------------------- #

def launch_app():
    CSS = """
html, body, .gr-app {height:100%; margin:0}
.top-row {height:100%; align-items:stretch;}
.gr-block {margin-bottom:0}
#rhs_col {display:flex; flex-direction:column; flex:1 1 auto;}
#product_gallery {height:auto !important;}
"""
    with gr.Blocks(css=CSS) as demo:
        # --------------------------------------------------------------
        # Persistent state
        # --------------------------------------------------------------
        chat_hist     = gr.State([])      # [(user, bot), …]
        prod_state    = gr.State([])      # current Product list (what is visible)
        sel_idx_state = gr.State(-1)      # selected gallery index
        precomputed   = gr.State([])      # full Product list (for “load more”)

        # --------------------------------------------------------------
        # Layout
        # --------------------------------------------------------------
        with gr.Row(elem_classes=["top-row"]):
            # -------- LEFT --------------------------------------------------
            with gr.Column(scale=3):
                gr.Markdown("### 1️⃣ Upload")
                img_in  = gr.Image(type="filepath", label="Inspiration")
                sub_btn = gr.Button("Get recommendations", variant="primary")

                gr.Markdown("### 2️⃣ Chat")
                chat     = gr.Chatbot(type="tuples")
                user_txt = gr.Textbox(label="Your message")

            # -------- RIGHT -------------------------------------------------
            with gr.Column(scale=2, elem_id="rhs_col"):
                gr.Markdown("### Recommendations")
                gallery_box = gr.Gallery(
                    label="Products",
                    elem_id="product_gallery",
                    columns=4,
                    object_fit="contain",
                    interactive=True,
                )
                load_more_btn = gr.Button("Load more 🔄", variant="secondary")

                with gr.Group(visible=False) as detail_box:
                    gr.Markdown("#### Product details")
                    d_img   = gr.Image()
                    d_price = gr.HTML()
                    with gr.Row():
                        buy_btn  = gr.Button("Buy now 💳",   variant="primary")
                        less_btn = gr.Button("Recommend less 👎", variant="secondary")
                        more_btn = gr.Button("Recommend more 👍",  variant="secondary")

        # --------------------------------------------------------------
        # Helper
        # --------------------------------------------------------------
        def _as_gallery(prods):
            """Convert List[Product] → List[(img, caption)] for <Gallery>"""
            return [(p.image, f"₹{p.price}") for p in prods]

        # --------------------------------------------------------------
        # Callbacks
        # --------------------------------------------------------------
        def _submit(img, hist):
            prods, reply = generate_initial_recommendations(img, n=N)
            hist = hist or []
            hist.append(("", reply))
            return (
                hist,                       # chat
                hist,                       # chat_hist (state)
                prods[:8],                  # prod_state
                -1,                         # sel_idx_state
                gr.update(visible=False),   # detail_box
                prods,                      # precomputed
                gr.update(value=_as_gallery(prods[:8]))  # gallery
            )

        sub_btn.click(
            _submit,
            inputs=[img_in, chat_hist],
            outputs=[chat, chat_hist, prod_state,
                     sel_idx_state, detail_box, precomputed, gallery_box],
        )

        def _chat(msg, hist, prods):
            if not msg:
                # nothing typed – just keep UI steady
                return (
                    gr.update(), hist, prods, "", -1,
                    gr.update(),   prods,
                    gr.update(value=_as_gallery(prods))
                )

            new_prods, reply = refine_recommendations(msg, hist, prods)
            hist.append((msg, reply))
            return (
                hist, hist,
                new_prods[:len(prods)],     # keep gallery size steady
                "", -1,
                gr.update(visible=False),
                new_prods,
                gr.update(value=_as_gallery(new_prods[:len(prods)]))
            )

        user_txt.submit(
            _chat,
            inputs=[user_txt, chat_hist, prod_state],
            outputs=[chat, chat_hist, prod_state,
                     user_txt, sel_idx_state, detail_box,
                     precomputed, gallery_box],
        )

        def _show(evt: gr.SelectData, prods):
            idx = evt.index if evt else None
            if idx is None or idx >= len(prods):
                return gr.update(), gr.update(), gr.update(visible=False), -1, gr.update()
            
            p = prods[idx]
            return (
                p.image,
                f"<h3>Price: {p.price}</h3>",
                gr.update(visible=True),
                idx,
                gr.update(value="Buy now 💳", link=p.link)
            )

        gallery_box.select(
            _show,
            inputs=[prod_state],
            outputs=[d_img, d_price, detail_box, sel_idx_state, buy_btn]  # ← buy_btn added
        )


        def _recommend_less(idx, hist, prods):
            if idx is None or idx < 0 or idx >= len(prods):
                return gr.update(), hist, prods, gr.update(), prods, gr.update(value=_as_gallery(prods))
            new_prods, bot_reply = recommend_less(idx, prods)
            hist.append((f"Recommend item {idx+1} less", bot_reply))
            return (
                hist, hist,
                new_prods[:len(prods)],
                gr.update(visible=False),
                new_prods,
                gr.update(value=_as_gallery(new_prods[:len(prods)])),
            )

        less_btn.click(
            _recommend_less,
            inputs=[sel_idx_state, chat_hist, prod_state],
            outputs=[chat, chat_hist, prod_state,
                     detail_box, precomputed, gallery_box],
        )

        def _recommend_more(idx, hist, prods):
            if idx is None or idx < 0 or idx >= len(prods):
                return gr.update(), hist, prods, gr.update(), prods, gr.update(value=_as_gallery(prods))
            new_prods, bot_reply = recommend_more(idx, prods)
            hist.append((f"Recommend item {idx+1} more", bot_reply))
            return (
                hist, hist,
                new_prods[:len(prods)],
                gr.update(visible=False),
                new_prods,
                gr.update(value=_as_gallery(new_prods[:len(prods)])),
            )

        more_btn.click(
            _recommend_more,
            inputs=[sel_idx_state, chat_hist, prod_state],
            outputs=[chat, chat_hist, prod_state,
                     detail_box, precomputed, gallery_box],
        )

        def _load_more(precomputed, prods, hist):
            if not precomputed:
                hist.append(("Load more", "Please upload an image first."))
                return hist, hist, prods, gr.update(value=_as_gallery(prods))

            if len(prods) + 8 > len(precomputed):
                hist.append(("Load more", "No more items available."))
                return hist, hist, prods, gr.update(value=_as_gallery(prods))

            new_view = precomputed[:len(prods) + 8]
            hist.append(("Load more", "Here you go!"))
            return hist, hist, new_view, gr.update(value=_as_gallery(new_view))

        load_more_btn.click(
            _load_more,
            inputs=[precomputed, prod_state, chat_hist],
            outputs=[chat, chat_hist, prod_state, gallery_box],
        )

    demo.queue().launch()


if __name__ == "__main__":
    launch_app()