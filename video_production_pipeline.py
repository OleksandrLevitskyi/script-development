#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ULTIMATE VIDEO PRODUCTION PIPELINE v2.0
Максимальное качество с цветными субтитрами и точной синхронизацией
Комбинированный скрипт: слайдшоу + озвучка + субтитры = готовое видео
"""

import os
import sys
import json
import time
import math
import random
import asyncio
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import logging

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Core processing imports
import cv2
import numpy as np
import whisper
import edge_tts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MotionEffects:
    """Плавные эффекты движения для слайдшоу"""
    
    @staticmethod
    def ease_in_out_sine(t: float) -> float:
        return -(math.cos(math.pi * t) - 1) / 2
    
    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2
    
    @staticmethod
    def ease_in_out_quart(t: float) -> float:
        if t < 0.5:
            return 8 * t * t * t * t
        return 1 - pow(-2 * t + 2, 4) / 2

class HighQualityImageProcessor:
    """Высококачественный обработчик изображений"""
    
    def __init__(self, width=1920, height=1080, quality_mode="high"):
        self.width = width
        self.height = height
        
        # Новые параметры для максимального качества
        if quality_mode == "high":
            self.quality_reduction = 0.6  # Улучшенное качество
            self.blur_radius = 30         # Больше блюра для плавности
        else:  # fast mode
            self.quality_reduction = 0.3
            self.blur_radius = 2
        
        # CUDA support
        self.cuda_available = False
        try:
            self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            if self.cuda_available:
                logger.info("✅ CUDA acceleration enabled")
        except:
            logger.info("ℹ️ Using CPU processing")
    
    def load_image_files(self, img_folder: Path):
        """Загрузка всех изображений из папки"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        
        image_files = []
        for ext in extensions:
            image_files.extend(img_folder.glob(f'*{ext}'))
            image_files.extend(img_folder.glob(f'*{ext.upper()}'))
        
        return sorted([str(f) for f in image_files])
    
    def extend_image_list(self, image_files: list, target_duration: float, fps: int = 30):
        """Расширение списка изображений для достижения нужной длительности"""
        total_frames_needed = int(target_duration * fps)
        slides_needed = max(10, total_frames_needed // (fps * 8))  # Минимум 10 слайдов, 8 сек на слайд
        
        if len(image_files) >= slides_needed:
            return image_files[:slides_needed]
        
        # Переиспользуем изображения в новом порядке
        extended_list = []
        original_count = len(image_files)
        
        for i in range(slides_needed):
            if i < original_count:
                extended_list.append(image_files[i])
            else:
                # Переиспользуем в случайном порядке
                random_index = random.randint(0, original_count - 1)
                extended_list.append(image_files[random_index])
        
        return extended_list
    
    def preprocess_image(self, image_path: str):
        """Высококачественная предобработка изображения"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            
            # Уменьшение для ускорения с сохранением качества
            height, width = img.shape[:2]
            new_width = int(width * self.quality_reduction)
            new_height = int(height * self.quality_reduction)
            
            img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Улучшенный блюр для плавности
            if self.blur_radius > 0:
                kernel_size = self.blur_radius * 2 + 1
                img_resized = cv2.GaussianBlur(img_resized, (kernel_size, kernel_size), 0)
            
            # Финальное масштабирование с лучшей интерполяцией
            img_final = cv2.resize(img_resized, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            
            return img_final
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def apply_motion_effect(self, img, effect_type: str, progress: float):
        """Применение плавных зум-эффектов с увеличенным качеством"""
        height, width = img.shape[:2]
        smooth_progress = MotionEffects.ease_in_out_sine(progress)
        
        # Ускоренный зум с плавными переходами
        max_zoom = 0.4
        zoom_curve = math.sin(smooth_progress * math.pi)
        scale = 1.0 + max_zoom * zoom_curve
        
        # Различные типы зумов
        centers = {
            "zoom_center": (width // 2, height // 2),
            "zoom_left": (width // 3, height // 2),
            "zoom_right": (width * 2 // 3, height // 2),
            "zoom_top": (width // 2, height // 3),
            "zoom_bottom": (width // 2, height * 2 // 3),
            "zoom_top_left": (width // 4, height // 4),
            "zoom_top_right": (width * 3 // 4, height // 4),
            "zoom_bottom_left": (width // 4, height * 3 // 4),
            "zoom_bottom_right": (width * 3 // 4, height * 3 // 4),
        }
        
        if effect_type == "static":
            breathing = 0.02 * math.sin(smooth_progress * math.pi * 2)
            scale = 1.0 + breathing
            center_x, center_y = width // 2, height // 2
        else:
            center_x, center_y = centers.get(effect_type, (width // 2, height // 2))
        
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
        
        try:
            if self.cuda_available:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                gpu_result = cv2.cuda.warpAffine(gpu_img, M, (width, height), 
                                               flags=cv2.INTER_CUBIC | cv2.WARP_FILL_OUTLIERS)
                result = gpu_result.download()
            else:
                result = cv2.warpAffine(img, M, (width, height), 
                                      flags=cv2.INTER_CUBIC | cv2.WARP_FILL_OUTLIERS)
        except:
            result = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_CUBIC)
        
        return result
    
    def create_transition_frames(self, img1, img2, transition_frames: int):
        """Создание максимально плавных переходных кадров"""
        frames = []
        
        for i in range(transition_frames):
            alpha = i / (transition_frames - 1) if transition_frames > 1 else 0
            smooth_alpha = MotionEffects.ease_in_out_quart(alpha)
            
            blended = cv2.addWeighted(img1, 1 - smooth_alpha, img2, smooth_alpha, 0)
            
            # Дополнительное сглаживание в середине перехода
            if transition_frames > 10 and i in range(transition_frames//3, 2*transition_frames//3):
                blended = cv2.GaussianBlur(blended, (3, 3), 0)
            
            frames.append(blended)
        
        return frames

class EnhancedTTSProcessor:
    """Улучшенный процессор текста в речь"""
    
    def __init__(self):
        self.voices = {
            'en': {
                'aria': 'en-US-AriaNeural',
                'jenny': 'en-US-JennyNeural',
                'michelle': 'en-US-MichelleNeural',
                'ana': 'en-US-AnaNeural',
                'emma': 'en-GB-SoniaNeural',
                'libby': 'en-GB-LibbyNeural',
                'mia': 'en-GB-MiaNeural',
            },
            'es': {
                'elvira': 'es-ES-ElviraNeural',
                'abril': 'es-ES-AbrilNeural',
                'delfina': 'es-MX-DaliaNeural',
                'renata': 'es-MX-RenataNeural',
                'sofia': 'es-CO-SalomeNeural',
                'ximena': 'es-CO-XimenaNeural',
                'camila': 'es-AR-ElenaNeural',
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Быстрое определение языка"""
        sample = text[:200].lower()
        
        # Испанские характерные элементы
        spanish_chars = ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'es', 'en', 'un', 'se', 'no', 
                        'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 
                        'del', 'los', 'las', 'una', 'este', 'esta']
        
        # Проверка на испанский
        if any(char in sample for char in spanish_chars):
            return 'es'
        
        words = sample.split()
        spanish_score = sum(1 for word in words if word in spanish_words)
        if spanish_score > len(words) * 0.25:
            return 'es'
        
        return 'en'
    
    async def text_to_speech(self, text: str, output_file: Path, config: dict):
        """Высококачественная генерация речи"""
        language = self.detect_language(text)
        voice_key = config['voices'][language]
        voice_name = self.voices[language][voice_key]
        
        # Скорость в Edge TTS формате
        speed_percent = int((config['speed'] - 1) * 100)
        rate_param = f"+{speed_percent}%" if speed_percent >= 0 else f"{speed_percent}%"
        
        logger.info(f"🎤 Generating speech: {language} - {voice_key} - {rate_param}")
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice_name,
            rate=rate_param
        )
        await communicate.save(str(output_file))
        
        return output_file

class AdvancedSlideshowGenerator:
    """Продвинутый генератор слайдшоу"""
    
    def __init__(self, config: dict):
        self.motion_effects = [
            "zoom_center", "zoom_left", "zoom_right", "zoom_top", "zoom_bottom",
            "zoom_top_left", "zoom_top_right", "zoom_bottom_left", "zoom_bottom_right", "static"
        ]
        self.processor = HighQualityImageProcessor(quality_mode=config.get('image_quality', 'high'))
    
    def get_audio_duration(self, audio_file: Path) -> float:
        """Получение длительности аудио файла"""
        try:
            result = subprocess.run([
                'ffprobe', '-i', str(audio_file), '-show_entries', 'format=duration',
                '-v', 'quiet', '-of', 'csv=p=0'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        return 300.0  # Fallback: 5 минут
    
    def create_slideshow(self, img_folder: Path, output_file: Path, target_duration: float, 
                        progress_callback=None):
        """Создание высококачественного синхронизированного слайдшоу"""
        try:
            # Загружаем изображения
            image_files = self.processor.load_image_files(img_folder)
            if not image_files:
                raise Exception("No images found")
            
            # Расширяем список под нужную длительность
            extended_images = self.processor.extend_image_list(image_files, target_duration)
            
            if progress_callback:
                progress_callback("🖼️ Preprocessing images with high quality...")
            
            # Предобработка изображений
            processed_images = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.processor.preprocess_image, img_path) 
                          for img_path in extended_images]
                
                for future in as_completed(futures):
                    img = future.result()
                    if img is not None:
                        processed_images.append(img)
            
            if not processed_images:
                raise Exception("Failed to process images")
            
            # Создание видео
            fps = 30
            width, height = 1920, 1080
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise Exception("Failed to create video writer")
            
            total_frames = int(target_duration * fps)
            frames_per_slide = total_frames // len(processed_images)
            
            if progress_callback:
                progress_callback("🎬 Creating high-quality slideshow...")
            
            frame_count = 0
            for i, img in enumerate(processed_images):
                if frame_count >= total_frames:
                    break
                
                effect_type = random.choice(self.motion_effects)
                
                # Основные кадры слайда
                for frame_num in range(frames_per_slide):
                    if frame_count >= total_frames:
                        break
                    
                    progress = frame_num / max(frames_per_slide - 1, 1)
                    frame = self.processor.apply_motion_effect(img, effect_type, progress)
                    out.write(frame)
                    frame_count += 1
                    
                    if progress_callback and frame_count % (fps * 5) == 0:  # Update every 5 seconds
                        percent = (frame_count / total_frames) * 100
                        progress_callback(f"🎬 Creating slideshow: {percent:.1f}%")
                
                # Плавные переходы между слайдами
                if i < len(processed_images) - 1 and frame_count < total_frames:
                    next_img = processed_images[i + 1]
                    next_effect = random.choice(self.motion_effects)
                    
                    # Последний кадр текущего слайда
                    final_frame = self.processor.apply_motion_effect(img, effect_type, 1.0)
                    # Первый кадр следующего слайда  
                    initial_next_frame = self.processor.apply_motion_effect(next_img, next_effect, 0.0)
                    
                    # Создаем плавный переход
                    transition_frames = int(1.2 * fps)
                    transitions = self.processor.create_transition_frames(
                        final_frame, initial_next_frame, transition_frames)
                    
                    for frame in transitions:
                        if frame_count >= total_frames:
                            break
                        out.write(frame)
                        frame_count += 1
            
            out.release()
            logger.info(f"✅ High-quality slideshow created: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Slideshow creation failed: {e}")
            return False

class PrecisionVideoMerger:
    """Высокоточное объединение видео, аудио и цветных субтитров"""
    
    def __init__(self):
        self.check_ffmpeg()
    
    def check_ffmpeg(self):
        """Проверка FFmpeg"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("FFmpeg not working")
            logger.info("✅ FFmpeg ready")
        except Exception as e:
            logger.error("❌ FFmpeg not found. Download from https://ffmpeg.org/")
            raise
    
    def merge_video_audio(self, video_file: Path, audio_file: Path, output_file: Path):
        """Объединение видео и аудио с точной синхронизацией"""
        cmd = [
            'ffmpeg', '-i', str(video_file), '-i', str(audio_file),
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
            '-map', '0:v:0', '-map', '1:a:0',
            '-avoid_negative_ts', 'make_zero',
            '-async', '1',  # улучшенная синхронизация
            '-y', str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def extract_audio_from_merged_video(self, video_path: Path, audio_output_path: Path):
        """Извлечение аудио из объединенного видео для точной синхронизации"""
        try:
            logger.info(f"🎵 Extracting audio from merged video: {video_path.name}")
            
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn',  # без видео
                '-acodec', 'pcm_s16le',  # несжатый формат для Whisper
                '-ar', '16000',  # частота для Whisper
                '-ac', '1',      # моно
                '-y', str(audio_output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Audio extracted from merged video")
                return True
            else:
                logger.error(f"❌ Audio extraction error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Audio extraction failed: {e}")
            return False
    
    def generate_colored_subtitles(self, audio_file: Path, subtitle_file: Path, config: dict):
        """Генерация цветных субтитров с word-level timestamps"""
        try:
            logger.info("🤖 Loading Whisper model...")
            model = whisper.load_model("base")
            
            logger.info("🎤 Transcribing with word-level timestamps...")
            result = model.transcribe(
                str(audio_file), 
                fp16=False, 
                verbose=False,
                word_timestamps=True  # Включаем word-level timestamps
            )
            
            # Конвертация в ASS с цветами
            ass_content = self.whisper_to_colored_ass(result, config)
            
            with open(subtitle_file, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            logger.info(f"✅ Colored subtitles generated: {subtitle_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Subtitle generation failed: {e}")
            return False
    
    def whisper_to_colored_ass(self, result, config: dict):
        """Конвертация в ASS с настраиваемыми цветами"""
        # Получаем цвета из конфига
        primary_color = config.get('subtitle_colors', {}).get('primary', '&H00FFFF&')    # желтый
        secondary_color = config.get('subtitle_colors', {}).get('secondary', '&HFFFFFF&') # белый
        
        # ASS заголовок
        ass_content = """[Script Info]
Title: Generated Colored Subtitles
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,28,&Hffffff,&Hffffff,&H000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,5,10,10,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text

"""
        
        subtitle_offset = config.get('subtitle_offset', 0.0)
        
        for segment in result['segments']:
            # Используем word-level данные если доступны
            if 'words' in segment and segment['words']:
                words = segment['words']
                word_chunks = self.group_words_into_chunks(words, max_words=4)
                
                for chunk in word_chunks:
                    if not chunk:
                        continue
                    
                    start_time = chunk[0]['start'] + subtitle_offset
                    end_time = chunk[-1]['end'] + subtitle_offset
                    
                    # Форматируем текст с чередующимися цветами
                    text = self.format_colored_text(chunk, primary_color, secondary_color)
                    
                    if text.strip():
                        start_ass = self.seconds_to_ass_time(max(0, start_time))
                        end_ass = self.seconds_to_ass_time(max(0.1, end_time))
                        
                        ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}\n"
            
            else:
                # Fallback к сегментам без word-level данных
                start_time = segment['start'] + subtitle_offset
                end_time = segment['end'] + subtitle_offset
                text = segment['text'].strip()
                
                if text:
                    formatted_text = self.format_colored_text_from_string(
                        text, primary_color, secondary_color)
                    start_ass = self.seconds_to_ass_time(max(0, start_time))
                    end_ass = self.seconds_to_ass_time(max(0.1, end_time))
                    
                    ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{formatted_text}\n"
        
        return ass_content
    
    def format_colored_text(self, word_chunk, primary_color, secondary_color):
        """Форматирование текста с чередующимися цветами"""
        formatted_words = []
        
        for i, word_data in enumerate(word_chunk):
            word = word_data['word'].strip().upper()
            
            if i % 2 == 0:  # четные позиции - первый цвет
                formatted_words.append(f"{{\\c{primary_color}}}{word}{{\\r}}")
            else:  # нечетные позиции - второй цвет
                formatted_words.append(f"{{\\c{secondary_color}}}{word}{{\\r}}")
        
        return " ".join(formatted_words)
    
    def format_colored_text_from_string(self, text, primary_color, secondary_color):
        """Форматирование обычного текста с чередующимися цветами"""
        words = text.strip().split()
        formatted_words = []
        
        for i, word in enumerate(words):
            word = word.upper()
            
            if i % 2 == 0:  # четные позиции - первый цвет
                formatted_words.append(f"{{\\c{primary_color}}}{word}{{\\r}}")
            else:  # нечетные позиции - второй цвет
                formatted_words.append(f"{{\\c{secondary_color}}}{word}{{\\r}}")
        
        return " ".join(formatted_words)
    
    def group_words_into_chunks(self, words, max_words=4):
        """Группировка слов в чанки"""
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            if len(current_chunk) >= max_words:
                chunks.append(current_chunk)
                current_chunk = []
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def seconds_to_ass_time(self, seconds):
        """Конвертация секунд в ASS формат"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def add_colored_subtitles_to_video(self, video_file: Path, subtitle_file: Path, output_file: Path):
        """Добавление цветных субтитров к видео"""
        subtitle_path_escaped = str(subtitle_file).replace('\\', '\\\\').replace(':', '\\:')
        
        cmd = [
            'ffmpeg', '-i', str(video_file),
            '-vf', f"ass='{subtitle_path_escaped}'",
            '-c:a', 'copy', '-y', str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

class UltimateVideoProductionPipeline:
    """Продвинутый пайплайн производства видео с максимальным качеством"""
    
    def __init__(self):
        self.tts_processor = EnhancedTTSProcessor()
        self.video_merger = PrecisionVideoMerger()
        self.progress_callback = None
        
    def find_video_folders(self, root_path: Path):
        """Поиск всех папок video_X"""
        video_folders = []
        
        for folder in root_path.iterdir():
            if folder.is_dir() and folder.name.startswith('video_'):
                try:
                    num = int(folder.name.split('_')[1])
                    video_folders.append((num, folder))
                except (IndexError, ValueError):
                    continue
        
        video_folders.sort(key=lambda x: x[0])
        return [folder for _, folder in video_folders]
    
    def create_folder_structure(self, video_folder: Path):
        """Создание расширенной структуры папок"""
        subfolders = ['img', 'text', 'voice', 'subtitles', 'slideshow', 'output']
        for subfolder in subfolders:
            (video_folder / subfolder).mkdir(exist_ok=True)
        
        # Создаем расширенный config.json
        config_file = video_folder / 'config.json'
        if not config_file.exists():
            config = {
                'voices': {'en': 'aria', 'es': 'elvira'},
                'speed': 0.95,
                'slideshow_effects': True,
                'subtitle_style': 'colorful',
                'subtitle_offset': 0.0,
                'word_timestamps': True,
                'image_quality': 'high',  # high/fast
                'subtitle_colors': {
                    'primary': '&H00FFFF&',    # желтый
                    'secondary': '&HFFFFFF&'   # белый
                },
                'blur_radius': 30,
                'quality_reduction': 0.6
            }
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
    
    def validate_folder(self, video_folder: Path):
        """Валидация папки перед обработкой"""
        img_folder = video_folder / 'img'
        text_folder = video_folder / 'text'
        
        # Проверяем наличие изображений
        processor = HighQualityImageProcessor()
        image_files = processor.load_image_files(img_folder)
        if not image_files:
            return False, "No images found in img/ folder"
        
        # Проверяем наличие текстового файла
        text_files = list(text_folder.glob('*.txt'))
        if not text_files:
            return False, "No text file found in text/ folder"
        
        return True, "OK"
    
    def process_single_video(self, video_folder: Path):
        """Обработка одной папки video_X с максимальным качеством"""
        try:
            folder_name = video_folder.name
            logger.info(f"🎬 Processing: {folder_name}")
            
            if self.progress_callback:
                self.progress_callback(f"📁 Processing {folder_name}...")
            
            # Создаем структуру папок
            self.create_folder_structure(video_folder)
            
            # Валидация
            is_valid, error_msg = self.validate_folder(video_folder)
            if not is_valid:
                logger.error(f"❌ {folder_name}: {error_msg}")
                return False
            
            # Загружаем расширенную конфигурацию
            config_file = video_folder / 'config.json'
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Пути к файлам
            text_file = next((video_folder / 'text').glob('*.txt'))
            voice_file = video_folder / 'voice' / f'{folder_name}_voice.mp3'
            slideshow_file = video_folder / 'slideshow' / f'{folder_name}_slideshow.mp4'
            subtitle_file = video_folder / 'subtitles' / f'{folder_name}_subtitles.ass'  # ASS формат
            temp_video = video_folder / 'output' / f'{folder_name}_temp.mp4'
            temp_audio = video_folder / 'output' / f'{folder_name}_temp.wav'
            final_video = video_folder / 'output' / f'{folder_name}_final.mp4'
            
            # Шаг 1: Генерация высококачественной озвучки
            if self.progress_callback:
                self.progress_callback(f"🎤 {folder_name}: Generating high-quality voice...")
            
            with open(text_file, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.tts_processor.text_to_speech(text_content, voice_file, config)
                )
            finally:
                loop.close()
            
            # Получаем длительность озвучки
            audio_duration = self.get_audio_duration(voice_file)
            logger.info(f"📏 Audio duration: {audio_duration:.1f} seconds")
            
            # Шаг 2: Создание высококачественного слайдшоу
            if self.progress_callback:
                self.progress_callback(f"🎬 {folder_name}: Creating high-quality slideshow...")
            
            slideshow_gen = AdvancedSlideshowGenerator(config)
            success = slideshow_gen.create_slideshow(
                video_folder / 'img',
                slideshow_file,
                audio_duration,
                lambda msg: self.progress_callback(f"{folder_name}: {msg}") if self.progress_callback else None
            )
            
            if not success:
                logger.error(f"❌ Failed to create slideshow for {folder_name}")
                return False
            
            # Шаг 3: Точное объединение видео и аудио
            if self.progress_callback:
                self.progress_callback(f"🔗 {folder_name}: Precision merging video and audio...")
            
            if not self.video_merger.merge_video_audio(slideshow_file, voice_file, temp_video):
                logger.error(f"❌ Failed to merge video and audio for {folder_name}")
                return False
            
            # Шаг 4: Извлечение аудио для точной синхронизации
            if self.progress_callback:
                self.progress_callback(f"🎵 {folder_name}: Extracting audio for precise sync...")
            
            if not self.video_merger.extract_audio_from_merged_video(temp_video, temp_audio):
                logger.error(f"❌ Failed to extract audio for {folder_name}")
                return False
            
            # Шаг 5: Генерация цветных субтитров с word-level timestamps
            if self.progress_callback:
                self.progress_callback(f"🌈 {folder_name}: Generating colored subtitles...")
            
            if not self.video_merger.generate_colored_subtitles(temp_audio, subtitle_file, config):
                logger.error(f"❌ Failed to generate colored subtitles for {folder_name}")
                return False
            
            # Шаг 6: Добавление цветных субтитров к финальному видео
            if self.progress_callback:
                self.progress_callback(f"✨ {folder_name}: Adding colored subtitles...")
            
            if not self.video_merger.add_colored_subtitles_to_video(temp_video, subtitle_file, final_video):
                logger.error(f"❌ Failed to add colored subtitles for {folder_name}")
                return False
            
            # Очистка временных файлов
            if temp_audio.exists():
                temp_audio.unlink()
            
            logger.info(f"✅ {folder_name}: Complete! Final video: {final_video}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical error processing {video_folder.name}: {e}")
            return False
    
    def get_audio_duration(self, audio_file: Path) -> float:
        """Получение длительности аудио файла"""
        try:
            result = subprocess.run([
                'ffprobe', '-i', str(audio_file), '-show_entries', 'format=duration',
                '-v', 'quiet', '-of', 'csv=p=0'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        return 300.0  # Fallback
    
    def process_all_videos(self, root_path: Path, progress_callback=None):
        """Обработка всех видео папок с максимальным качеством"""
        self.progress_callback = progress_callback
        
        video_folders = self.find_video_folders(root_path)
        
        if not video_folders:
            logger.error("❌ No video_X folders found")
            return False
        
        logger.info(f"📁 Found {len(video_folders)} video folders")
        
        success_count = 0
        total_count = len(video_folders)
        
        for i, video_folder in enumerate(video_folders, 1):
            if progress_callback:
                progress_callback(f"🎯 Processing {i}/{total_count}: {video_folder.name}")
            
            success = self.process_single_video(video_folder)
            if success:
                success_count += 1
            
            overall_progress = (i / total_count) * 100
            if progress_callback:
                progress_callback(f"📊 Overall progress: {overall_progress:.1f}% ({i}/{total_count})")
        
        logger.info(f"🎉 Processing complete! Success: {success_count}/{total_count}")
        return success_count == total_count

class AdvancedVideoProductionGUI:
    """Продвинутый графический интерфейс"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎬 Ultimate Video Production Pipeline v2.0 - Maximum Quality")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        self.pipeline = UltimateVideoProductionPipeline()
        self.root_path = tk.StringVar()
        self.is_processing = False
        
        self.create_widgets()
        
    def create_widgets(self):
        """Создание продвинутого интерфейса"""
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="🎬 Ultimate Video Production Pipeline v2.0", 
                               font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="🌈 High Quality • Colored Subtitles • Precise Sync", 
                                  font=("Arial", 12, "italic"))
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Выбор корневой папки
        folder_frame = ttk.LabelFrame(main_frame, text="📁 Root Folder Selection", padding="10")
        folder_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(folder_frame, text="Select folder containing video_1, video_2, etc.:").grid(row=0, column=0, sticky=tk.W)
        
        path_frame = ttk.Frame(folder_frame)
        path_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.path_entry = ttk.Entry(path_frame, textvariable=self.root_path, width=80)
        self.path_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(path_frame, text="Browse", command=self.browse_folder).grid(row=0, column=1)
        ttk.Button(path_frame, text="Scan", command=self.scan_folders).grid(row=0, column=2, padx=(5, 0))
        
        path_frame.columnconfigure(0, weight=1)
        
        # Список найденных папок
        list_frame = ttk.LabelFrame(main_frame, text="📋 Found Video Folders", padding="10")
        list_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        
        columns = ('folder', 'images', 'text', 'config', 'status')
        self.folder_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        self.folder_tree.heading('folder', text='Folder')
        self.folder_tree.heading('images', text='Images')
        self.folder_tree.heading('text', text='Text Files')
        self.folder_tree.heading('config', text='Config')
        self.folder_tree.heading('status', text='Status')
        
        self.folder_tree.column('folder', width=120)
        self.folder_tree.column('images', width=80)
        self.folder_tree.column('text', width=80)
        self.folder_tree.column('config', width=80)
        self.folder_tree.column('status', width=200)
        
        tree_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.folder_tree.yview)
        self.folder_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.folder_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Информация о функциях
        info_frame = ttk.LabelFrame(main_frame, text="✨ Features", padding="10")
        info_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        features_text = """🎨 High-Quality Processing • 🌈 Colored Subtitles (Yellow/White) • ⚡ Word-Level Timestamps
🔄 Precise Audio Sync • 📏 Duration Matching • 🎬 Enhanced Motion Effects • ⚙️ Full Config Control"""
        
        ttk.Label(info_frame, text=features_text, justify=tk.CENTER, 
                 font=("Arial", 10)).grid(row=0, column=0)
        
        # Кнопки управления
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=(0, 15))
        
        self.start_button = ttk.Button(button_frame, text="🚀 Start Maximum Quality Production", 
                                      command=self.start_production)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="🔄 Refresh", command=self.scan_folders).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="📋 View Logs", command=self.show_logs).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="⚙️ Config Help", command=self.show_config_help).pack(side=tk.LEFT)
        
        # Прогресс
        progress_frame = ttk.LabelFrame(main_frame, text="📊 Progress", padding="10")
        progress_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_var = tk.StringVar(value="Ready for maximum quality production...")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        progress_frame.columnconfigure(0, weight=1)
        
        # Логи
        log_frame = ttk.LabelFrame(main_frame, text="📝 Activity Log", padding="5")
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Настройка весов для адаптивности
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        main_frame.rowconfigure(7, weight=1)
        folder_frame.columnconfigure(0, weight=1)
        
        # Начальное сообщение
        self.add_log("🚀 Ultimate Video Production Pipeline v2.0 initialized")
        self.add_log("🌈 Features: High Quality • Colored Subtitles • Precise Sync")
        self.add_log("📁 Select root folder and click 'Scan' to find video folders")
    
    def browse_folder(self):
        """Выбор корневой папки"""
        folder = filedialog.askdirectory(title="Select root folder containing video_X folders")
        if folder:
            self.root_path.set(folder)
            self.scan_folders()
    
    def scan_folders(self):
        """Сканирование папок video_X"""
        if not self.root_path.get():
            messagebox.showerror("Error", "Please select a root folder first")
            return
        
        root_path = Path(self.root_path.get())
        if not root_path.exists():
            messagebox.showerror("Error", "Selected folder does not exist")
            return
        
        # Очищаем список
        for item in self.folder_tree.get_children():
            self.folder_tree.delete(item)
        
        # Поиск папок
        video_folders = self.pipeline.find_video_folders(root_path)
        
        if not video_folders:
            self.add_log("❌ No video_X folders found in selected directory")
            return
        
        self.add_log(f"📁 Found {len(video_folders)} video folders")
        
        # Заполняем список
        for video_folder in video_folders:
            # Подсчитываем файлы
            img_folder = video_folder / 'img'
            text_folder = video_folder / 'text'
            config_file = video_folder / 'config.json'
            
            processor = HighQualityImageProcessor()
            img_count = len(processor.load_image_files(img_folder)) if img_folder.exists() else 0
            text_count = len(list(text_folder.glob('*.txt'))) if text_folder.exists() else 0
            has_config = "✅" if config_file.exists() else "❌"
            
            # Определяем статус
            if img_count > 0 and text_count > 0:
                status = "✅ Ready for high-quality processing"
            elif img_count == 0 and text_count == 0:
                status = "❌ Missing img & text folders"
            elif img_count == 0:
                status = "❌ Missing images"
            elif text_count == 0:
                status = "❌ Missing text file"
            else:
                status = "⚠️ Check configuration"
            
            self.folder_tree.insert('', 'end', values=(
                video_folder.name,
                f"{img_count} files",
                f"{text_count} files",
                has_config,
                status
            ))
        
        self.add_log(f"✅ Scan complete: {len(video_folders)} folders ready for processing")
    
    def start_production(self):
        """Запуск производства видео максимального качества"""
        if self.is_processing:
            messagebox.showwarning("Warning", "Production is already running!")
            return
        
        if not self.root_path.get():
            messagebox.showerror("Error", "Please select a root folder first")
            return
        
        # Проверяем наличие папок для обработки
        ready_count = 0
        for item in self.folder_tree.get_children():
            status = self.folder_tree.item(item)['values'][4]
            if "✅ Ready" in status:
                ready_count += 1
        
        if ready_count == 0:
            messagebox.showerror("Error", "No folders are ready for processing!\nMake sure each video_X folder has images and text files.")
            return
        
        # Подтверждение
        result = messagebox.askyesno("Confirm Maximum Quality Production", 
                                   f"Start high-quality processing of {ready_count} video folders?\n\n" +
                                   "Features enabled:\n" +
                                   "🎨 Enhanced image quality\n" +
                                   "🌈 Colored subtitles (Yellow/White)\n" +
                                   "⚡ Word-level timestamps\n" +
                                   "🔄 Precise audio synchronization\n\n" +
                                   "This may take a while but will produce the best results...")
        if not result:
            return
        
        self.is_processing = True
        self.start_button.config(text="🔄 Processing Maximum Quality...", state="disabled")
        self.progress_bar.start()
        
        self.add_log(f"🚀 Starting maximum quality production of {ready_count} videos...")
        
        # Запускаем обработку в отдельном потоке
        def run_production():
            try:
                root_path = Path(self.root_path.get())
                success = self.pipeline.process_all_videos(root_path, self.update_progress)
                
                self.root.after(0, self.production_complete, success)
                
            except Exception as e:
                logger.error(f"Production error: {e}")
                self.root.after(0, self.production_error, str(e))
        
        thread = threading.Thread(target=run_production, daemon=True)
        thread.start()
    
    def update_progress(self, message: str):
        """Обновление прогресса"""
        self.root.after(0, lambda: self.progress_var.set(message))
        self.root.after(0, lambda: self.add_log(message))
    
    def production_complete(self, success: bool):
        """Завершение производства"""
        self.is_processing = False
        self.progress_bar.stop()
        self.start_button.config(text="🚀 Start Maximum Quality Production", state="normal")
        
        if success:
            self.progress_var.set("🎉 All high-quality videos completed successfully!")
            self.add_log("🎉 Maximum quality production completed successfully!")
            messagebox.showinfo("Success", "All videos have been processed with maximum quality!\n\n" +
                              "Features applied:\n" +
                              "🎨 Enhanced image processing\n" +
                              "🌈 Colored subtitles with precise timing\n" +
                              "🔄 Perfect audio synchronization")
        else:
            self.progress_var.set("⚠️ Production completed with some errors")
            self.add_log("⚠️ Production completed with some errors - check logs")
            messagebox.showwarning("Warning", "Production completed but some videos failed.\nCheck the activity log for details.")
        
        # Обновляем список папок
        self.scan_folders()
    
    def production_error(self, error_msg: str):
        """Обработка ошибки производства"""
        self.is_processing = False
        self.progress_bar.stop()
        self.start_button.config(text="🚀 Start Maximum Quality Production", state="normal")
        self.progress_var.set("❌ Production failed")
        self.add_log(f"❌ Production failed: {error_msg}")
        messagebox.showerror("Error", f"Production failed:\n{error_msg}")
    
    def show_logs(self):
        """Показать окно с логами"""
        log_window = tk.Toplevel(self.root)
        log_window.title("📋 Detailed Production Logs")
        log_window.geometry("900x700")
        
        log_text = tk.Text(log_window, wrap=tk.WORD, font=("Consolas", 10))
        log_scroll = ttk.Scrollbar(log_window, orient=tk.VERTICAL, command=log_text.yview)
        log_text.configure(yscrollcommand=log_scroll.set)
        
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Получаем содержимое основного лога
        log_content = self.log_text.get(1.0, tk.END)
        log_text.insert(1.0, log_content)
        log_text.config(state=tk.DISABLED)
    
    def show_config_help(self):
        """Показать справку по конфигурации"""
        help_window = tk.Toplevel(self.root)
        help_window.title("⚙️ Configuration Help")
        help_window.geometry("800x600")
        
        help_text = """
📁 CONFIG.JSON STRUCTURE

Each video_X folder automatically gets a config.json file with these settings:

🎤 VOICE SETTINGS:
• "voices": {"en": "aria", "es": "elvira"} - Voice selection
• "speed": 0.95 - Speech speed (0.5-2.0)

🎨 VISUAL SETTINGS:
• "image_quality": "high" - Image processing quality
• "blur_radius": 30 - Motion blur amount
• "quality_reduction": 0.6 - Image compression balance

🌈 SUBTITLE SETTINGS:
• "subtitle_colors": {
    "primary": "&H00FFFF&",    # Yellow (ASS format)
    "secondary": "&HFFFFFF&"   # White (ASS format)
  }
• "subtitle_offset": 0.0 - Time shift in seconds (+/-)
• "word_timestamps": true - Word-level precision

⚙️ ADVANCED:
• "slideshow_effects": true - Enable motion effects
• "subtitle_style": "colorful" - Subtitle appearance

💡 COLOR CODES (ASS format):
• "&H0000FF&" - Red
• "&H00FF00&" - Green  
• "&H00FFFF&" - Yellow
• "&HFFFFFF&" - White
• "&HFF0000&" - Blue
• "&HFF00FF&" - Magenta
• "&HFFFF00&" - Cyan

🎯 TIPS:
- Edit config.json in each video_X folder for custom settings
- Restart processing after config changes
- Use negative subtitle_offset if subtitles appear too late
- Higher blur_radius = smoother motion (slower processing)
"""
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, font=("Consolas", 10), padx=10, pady=10)
        scroll_help = ttk.Scrollbar(help_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scroll_help.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_help.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert(1.0, help_text)
        text_widget.config(state=tk.DISABLED)
    
    def add_log(self, message: str):
        """Добавление сообщения в лог"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        
        # Ограничиваем размер лога
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.log_text.delete(1.0, "200.0")
    
    def run(self):
        """Запуск интерфейса"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")

def main():
    """Главная функция"""
    print("🎬 Ultimate Video Production Pipeline v2.0")
    print("🌈 Maximum Quality • Colored Subtitles • Precise Sync")
    print("="*60)
    
    # Проверка зависимостей
    try:
        import cv2
        import numpy as np
        import whisper
        import edge_tts
        print("✅ All dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Install required packages:")
        print("pip install opencv-python numpy openai-whisper edge-tts")
        return 1
    
    # Проверка FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("FFmpeg not working")
        print("✅ FFmpeg found")
    except:
        print("❌ FFmpeg not found!")
        print("📥 Download FFmpeg from: https://ffmpeg.org/download.html")
        print("⚙️ Add FFmpeg to your system PATH")
        return 1
    
    print("🚀 Starting Ultimate Video Production Pipeline v2.0...")
    print("🎯 Features: Enhanced Quality • Colored Subtitles • Word-Level Sync")
    
    # Запуск GUI
    app = AdvancedVideoProductionGUI()
    app.run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())