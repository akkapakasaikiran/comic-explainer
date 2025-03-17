import os
import json
import glob
import re
import shutil
import moviepy
from PIL import Image
from dotenv import load_dotenv

from google import genai
from google.genai import types
from io import BytesIO
import elevenlabs
import argparse


# Load environment variables
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Comic Generation Workflow Automation')
    parser.add_argument('--age', type=int, default=10, help='Target age of the audience')
    parser.add_argument('--theme', type=str, default='Pokemon', help='Theme of the comic story')
    parser.add_argument('--topic', type=str, default='gravity', help='Topic of the comic story')
    parser.add_argument('--output_dir', type=str, default='comic_project', help='Output directory for the project')
    return parser.parse_args()

class ComicWorkflowAutomation:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        self.elevenlabs_client = elevenlabs.client.ElevenLabs(api_key=self.elevenlabs_api_key)                 
        self.voice_ids = [voice.voice_id for voice in self.elevenlabs_client.voices.get_all().voices]

        
    def ensure_directories(self, output_dir):
        """Create necessary directories for the project"""
        self.output_dir = output_dir
        
        # create backup directory
        self.backup_dir = f"{output_dir}_backup"

        # backup the output directory
        if os.path.exists(self.output_dir):
            shutil.copytree(self.output_dir, self.backup_dir, dirs_exist_ok=True)
            
            # empty the output directory
            shutil.rmtree(self.output_dir)
            
        # create fresh output directories
        directories = ["story", "images", "audio", "final"]
        for dir in directories:
            path = os.path.join(self.output_dir, dir)
            os.makedirs(path)
          
          
    def call_gemini_api(self, prompt):
        """Call Gemini API with given prompt"""
        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
        )
        
        out = {'text': None, 'image': None}
        try:
            # print(f"Response: {response}")
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    out['text'] = part.text
                if part.inline_data is not None:
                    out['image'] = Image.open(BytesIO(part.inline_data.data))
        except Exception as e:
            print(f"Error processing Gemini API response: {e}")
            print(f'Prompt: {prompt}')
                
        return out
    
  
    def generate_story(self, age, theme, topic):
        """Generate comic story concept using Gemini"""
        
        prompt = f"""
        Create a comic story explaining {topic} to a {age}-year old kid.
        Theme: {theme}. The comic should feature at most two characters. 
        It should be fun, engaging, and educational.
        Each panel of the story should be easy to illustrate.
        
        For each panel, give me the following:
        - image_description: What is happening in the panel? I will use this as a prompt for text to image generative models. 
        - dialogues: What are the characters saying?
        
        Structure your response as a JSON object with the following schema:
            
        {{
            "comic_story": 
            [
                {{
                    "image_description": "Description of panel 1",
                    "dialogues": [
                        "character_1": "Dialogue for character1",
                        "character_2": "Dialogue for character2"
                    ]
                }},
                ...
            ]
        }}
        """
        
        attempts, max_num_attempts = 0, 5
        response = None
        while attempts < max_num_attempts:
            response = self.call_gemini_api(prompt)
            
            json_match = re.search(r'({[\s\S]*})', response["text"])
            if json_match:
                text_response = json_match.group(1)
                try:
                    comic_script = json.loads(text_response)
                    image_descriptions = [scene['image_description'] for scene in comic_script['comic_story']]
                    dialogues = [scene['dialogues'] for scene in comic_script['comic_story']]
                    
                    # save the comic script to a file
                    with open(os.path.join(self.output_dir, "story", "comic_script.json"), "w") as f:
                        json.dump(comic_script, f, indent=4)
                    
                    print("Comic story generated successfully!")
                    return image_descriptions, dialogues
                
                except json.JSONDecodeError:
                    print(f"Attempt {attempts + 1} failed: JSON decode error, retrying...")
            else:
                print(f"Attempt {attempts + 1} failed: No JSON match, retrying...")
            attempts += 1
        
        raise ValueError(f"Failed to generate story after {max_num_attempts} attempts")
            
    
    def generate_images(self, image_descriptions):
        """Generate images for each panel using Gemini"""
        
        panels = []
        for i, description in enumerate(image_descriptions):
            print(f"Generating image for panel {i+1}...")
           
            attempts, max_num_attempts = 0, 5
            image = None
            while attempts < max_num_attempts and image is None:
                prompt = "Generate an image: " + description
                response = self.call_gemini_api(prompt)
                image = response["image"]
                attempts += 1
                if image is None:
                    print(f"Attempt {attempts} failed, retrying...")
            
            if image is None:
                raise ValueError(f"Failed to generate image for panel {i+1} after {max_num_attempts} attempts")
            
            image_path = os.path.join(self.output_dir, "images", f"panel_{i+1}.png")
            image.save(image_path)
            
            panels.append({"image_path": image_path})
        
        return panels
    
    def generate_audio(self, dialogues):
        """ Generate audio for each panel using Eleven Labs API"""
        
        audio_dir = os.path.join(self.output_dir, "audio")
        
        # Extract characters and assign voices to them
        # TODO: Can use LLM to chose voices based on character descriptions
        characters = set()
        for dialogue in dialogues: 
            characters.update(dialogue.keys())                                  
        character_voices = dict(zip(characters, self.voice_ids))

        # Generate audio for each dialogue
        for i, dialogue in enumerate(dialogues):
            for character, line in dialogue.items():
                print(f"Generating audio for {character}...")
                audio = self.elevenlabs_client.text_to_speech.convert(
                    text=line,
                    voice_id=character_voices[character],
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                )      
                save_path = os.path.join(audio_dir, f"panel_{i+1}_{character}.mp3")
                elevenlabs.save(audio, save_path)
                
        print("Audio generation complete!")
        return        
    
    def assemble_video(self, output_path="output.mp4"):
        """
        Create a video by combining images with their corresponding audio files.
        """

        images_dir = os.path.join(self.output_dir, "images")
        audio_dir = os.path.join(self.output_dir, "audio")
        output_path = os.path.join(self.output_dir, "final", output_path)

        # Get all image files and sort them
        image_files = sorted(glob.glob(os.path.join(images_dir, "panel_*.png")), 
                            key=lambda x: int(x.split('panel_')[1].split('.')[0]))
        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")
        
        video_clips = []
        
        for image_path in image_files:
            # Extract panel number
            panel_num = os.path.basename(image_path).split('.')[0].split('_')[1]
            
            # Find all audio files for this panel
            panel_audio_files = glob.glob(os.path.join(audio_dir, f"panel_{panel_num}_*.mp3"))
            if not panel_audio_files:
                raise ValueError(f"No audio files found for panel {panel_num}")
        
            audio_clips = []
            current_duration = 0
            
            # Process each audio file for the current panel
            # BUG: we are ordering audios by character name. We should get the dialogue order
            # from LLM and store it. 
            for audio_path in sorted(panel_audio_files):
                audio_clip = moviepy.AudioFileClip(audio_path)
                # Set the start time for each audio clip
                audio_clips.append(audio_clip.with_start(current_duration))
                current_duration += audio_clip.duration

            # Combine all audio clips into one
            composite_audio_clip = moviepy.CompositeAudioClip(audio_clips)
        
            # Create an image clip with the combined audio and set its duration
            total_duration = current_duration
            img_clip = moviepy.ImageClip(image_path).with_duration(total_duration)
            img_clip = img_clip.with_audio(composite_audio_clip)
            video_clips.append(img_clip)
        
        # Concatenate all video clips
        final_clip = moviepy.concatenate_videoclips(video_clips, method="compose")
        
        # Write the result to a file
        final_clip.write_videofile(output_path, codec='libx264', fps=24)
        
        # Close all clips to free resources
        final_clip.close()
        for clip in video_clips:
            clip.close()
        
        return output_path

    
    def run_workflow(self, age, theme, topic, output_dir="comic_project"):
        """Run the entire workflow"""

        # Create necessary directories
        self.ensure_directories(output_dir)

        try:
            image_descriptions, dialogues = self.generate_story(age, theme, topic)
            self.generate_images(image_descriptions)        
            self.generate_audio(dialogues)        
            final_video = self.assemble_video()
            
            print(f"Workflow complete! Final video: {final_video}")
        except Exception as e:
            print(f"An error occurred: {e}")
            # Restore from backup if an error occurs
            if os.path.exists(self.backup_dir):
                shutil.rmtree(self.output_dir)
                shutil.copytree(self.backup_dir, self.output_dir, dirs_exist_ok=True)
            raise e
        finally:
            # Clean up the backup directory
            if os.path.exists(self.backup_dir):
                shutil.rmtree(self.backup_dir)

if __name__ == "__main__":
    args = parse_args()
    
    workflow = ComicWorkflowAutomation()
    workflow.run_workflow(args.age, args.theme, args.topic, args.output_dir)
