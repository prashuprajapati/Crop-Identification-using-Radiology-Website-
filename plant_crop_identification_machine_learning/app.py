import streamlit as st 
import tensorflow as tf
from PIL import Image,ImageOps
import cv2
import numpy as np 
import pickle
from streamlit_option_menu import option_menu

st.set_page_config(
        page_title='Crop identification',
        page_icon=":tada:",
        layout="wide",
        initial_sidebar_state="auto"
     )
with st.container():
    st.subheader("Hello, Welcome to user :wave:")
    st.title('Check your plant disease')

st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model():
    # model = tf.keras.models.load_model('plant_crop_identification.h5')
    model = pickle.load(open('model/svm_crop_identifiaction.pkl','rb'))
    #model = tf.keras.models.load_model('model/svm_crop_identifiaction')
    #model = pickle.load(open('cat_and_dog_classification.pkl', 'rb'))

    return model 

model = load_model()
with st.container():
    file = st.file_uploader("Please upload the plant image",type=["jpg","png"])


def extract_features(images):
    svm_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=[56,56,3])
    preprocessed_images = tf.keras.applications.resnet50.preprocess_input(images)
    features = svm_model.predict(preprocessed_images)
    # Flatten the 3-dimensional features to a 2-dimensional array
    features_flat = features.reshape(features.shape[0], -1)
    return features_flat


def import_and_predict(image_data,model):
    # size = (56,56)
    # image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    # img = np.asarray(image)
    # img = img.reshape(img.shape[0],-1)
    # img_reshape = img[np.newaxis,...]
    img_reshape = extract_features(image_data)
    prediction = model.predict(img_reshape)

    return prediction

if file is None: 
    st.text("pleae upload the image file")
else:
    with st.container():
        st.write("---")
        left_col,right_col= st.columns(2)
        with right_col:
                image = Image.open(file)
                st.image(image,use_column_width=False)

        predictions = import_and_predict(image,model)
        with left_col:
            class_names = ['Apple___Cedar_apple_rust', 'Apple___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            #class_names = ['cats', 'dogs']
            result = np.argmax(predictions)
            String = "The plant status : "+class_names[np.argmax(predictions)]  
            st.success(String)      

        with st.container():
            st.subheader("Solution of Disease")
            if result==0:
                st.subheader("what is the "+class_names[np.argmax(predictions)]+" ?")
                String = """Cedar apple rust is a disease caused by the fungal pathogen Gymnosporangium juniperi-virginianae, which requires two hosts: apple and red cedars / ornamental junipers to complete its lifecycle. On apple, the pathogen can infect leaves and fruit of susceptible cultivars and may cause premature defoliation if infection is severe."""
                st.write(String)
                dis_path = 'apple cader rust.jpg'
                image = Image.open(dis_path)
                st.image(image,use_column_width=False)
                st.subheader("How to Control Cedar Apple Rust")
                st.write(
                    """
                    The best way to control cedar apple rust is to prevent infection using a mixture of cultural methods and chemical treatments.
                    If you see the lesions on the apple leaves or fruit, it is too late to control the fungus. In that case, you should focus on purging infected leaves and fruit from around your tree.
                    """
                )
                spray_path ='Spraying-Apple-Trees-With-Copper-Fungicide.jpg' 
                image = Image.open(spray_path)
                st.image(image,use_column_width=False)
                st.write(
                    """
                    Don't plant junipers near rust-susceptible plants, which include both apples and crabapples. Consider resistant apple varieties, such as 'Freedom','Liberty','Redfree' or 'William's Pride.
                    And also destroy wild or unwanted apples, crabapples, or junipers, so they won't infect your apple tree.
                    """
                )
                st.subheader("Cultural Controls")
                st.write(
                    """
                    Since the juniper galls are the source of the spores that infect the apple trees, cutting them is a sound strategy if there aren’t too many of them.
                    While the spores can travel for miles, most of the ones that could infect your tree are within a few hundred feet.
                    The best way to do this is to prune the branches about 4-6 inches below the galls.
                    You will want to disinfect your pruning shears, so you don’t spread the infection. Dip them in 10% bleach or alcohol for at least 30 seconds between cuts.
                    """
                )
                st.subheader("Symptoms on Apples and Crabapples")
                st.write(
                    """
                    nstead of galls, infected apple and crabapple trees manifest circular yellow spots on the upper surface of their leaves soon after 
                    bloom. Later in the summer, brownish cylindrical tubes with hairs sticking out appear underneath the yellow spots, or on the twigs and fruit.
                    """
                )
                image = Image.open('Apple-tree-leaves-damaged-by-cedar-apple-rust.jpg')
                st.image(image,use_column_width=False)
                st.write(
                    """
                    These tubes produce the aeciospores that will complete the cycle by infecting the needles of junipers.
                    At the least, the infected fruit may be of marginal quality. Worst case scenario – they drop off the tree.
                    In addition, a severe infection can cause your tree to drop its leaves! If that happens for several years in a row, your apple tree could be in peril.
                    """
                )
                st.subheader("Fungicide Treatments")
                st.write(
                    """
                    - If your tree has a history of infection with cedar apple rust, you will want to get ahead of the infection and take preemptive measures.

                    - This is critical in the spring, when the juniper galls are releasing their spores.

                    - The time to treat your tree is between the pink stage of the blossoms (when the leaves are turning green) to the period when the petals drop.

                    - The most effective types of fungicides to use are those that inhibit fungal sterols. They are known as “SI,” or sterol inhibitors.

                    - In the old days, sprays for apple scab would also take care of cedar apple rust. However, this is no longer the case.

                    - The fungus that causes apple scab is now frequently resistant to the sterol-inhibiting fungicides, and manufacturers have moved on to using newer classes of fungicides. Only certain types of fungicides are effective.

                    - Extension agents at North Carolina State University attribute this trend to an increase in occurrences on apple trees in the state.

                    - Unfortunately, captan, the fungicide in many pre-mixed sprays for home fruit trees, does not work on on this particular fungal pathogen.

                    - Several extension agencies recommend that you use Immunox® to control cedar apple rust. It contains myclobutanil as its active ingredient.
                    """
                )
                image = Image.open('41x13ZYatvL.jpg')
                st.image(image,use_column_width=False)
                

            elif result==1:
                st.write("""
                The plant is health and no desease in the plant.
                """)
                
            elif result==2:
                st.subheader("what is the "+class_names[np.argmax(predictions)]+" ?")
                left_col,right_col = st.columns(2)
                
                with left_col:
                    st.write("""
                        Powdery mildew of sweet and sour cherry is caused by Podosphaera clandestina, an obligate biotrophic fungus.
                        Mid- and late-season sweet cherry (Prunus avium) cultivars are commonly affected, rendering them unmarketable 
                        due to the covering of white fungal growth on the cherry surface.Season long disease control of both leaves and 
                        fruit is critical to minimize overall disease pressure in the orchard and consequently to protect developing fruit
                        from accumulating spores on their surfaces.
                        """)
                with right_col:
                    powder_path = 'powdery_mildew_dease.png'
                    image = Image.open(powder_path)
                    st.image(image,use_column_width=False)
                st.subheader("identification")
                left,right = st.columns(2)
                with left :
                    st.write("""
                    Initial symptoms, often occurring 7 to 10 days after the onset of the first irrigation, 
                    are light roughly-circular, powdery looking patches on young, susceptible leaves (newly unfolded, 
                    and light green expanding leaves). Older leaves develop an age-related (ontogenic) resistance to
                    powdery mildew and are naturally more resistant to infection than younger leaves. Look for early 
                    leaf infections on root suckers, the interior of the canopy or the crotch of the tree where humidity 
                    is high. In contrast to other fungi, powdery mildews do not need free water to germinate but germination 
                    and fungal growth are favored by high humidity (Grove & Boal, 1991a). The disease is more likely to initiate
                    on the undersides (abaxial) of leaves (Fig. 2) but will occur on both sides at later stages. As the season
                    progresses and infection is spread by wind, leaves may become distorted, curling upward. 
                    Severe infections may cause leaves to pucker and twist. Newly developed leaves on new shoots 
                    become progressively smaller, are often pale and may be distorted.
                """)
                with right:
                    powder_path1 = 'powdery_mildew_identification.png'
                    image = Image.open(powder_path1)
                    st.image(image,use_column_width=False)
                    
            elif result==3:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==4:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==5:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==6:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==7:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==8:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==9:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==10:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==11:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==12:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==13:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==14:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==15:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==16:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==17:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==18:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==19:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==20:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==21:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==22:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==23:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==24:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==25:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==26:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==27:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==28:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==29:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==30:
                String = "The plant status : "+class_names[result]
                st.success(String)
            elif result==31:
                String = "The plant status : "+class_names[result]
                st.success(String) 