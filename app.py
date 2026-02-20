import os
# [중요] Mac 충돌 방지 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import gradio as gr
import numpy as np
import json
import tempfile
import traceback
import time
from pathlib import Path
from typing import Tuple, Optional

# image_to_vector.py가 같은 폴더에 있어야 합니다.
try:
    from image_to_vector import BridgeRecommender
except ImportError:
    print("⚠️ 경고: 'image_to_vector.py'를 찾을 수 없습니다.")
    class BridgeRecommender: pass 

class MusicRecommenderApp:
    def __init__(self, db_path="music_database.npy", meta_path="music_database_metadata.json"):
        self.db_path = Path(db_path)
        self.meta_path = Path(meta_path)
        
        if not self.db_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("❌ 데이터베이스 파일이 없습니다.")

        print("⏳ 데이터베이스 로딩 중...")
        self.vectors = np.load(self.db_path)
        
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-12)
        
        self.recommender = BridgeRecommender()
        print(f"✅ 앱 준비 완료: {len(self.metadata)}곡 로드됨.")
    
    # [UX 개선] progress=gr.Progress() 추가
    def recommend(self, image, topk: int = 5, progress=gr.Progress()):
        """
        이미지를 분석하고 음악을 추천합니다.
        """
        if image is None:
            return "⚠️ 이미지를 업로드해주세요!", "", None, ""
        
        temp_file_path = None
        
        try:
            # 1단계: 이미지 처리
            progress(0.1, desc="📸 이미지 읽는 중...")
            time.sleep(0.3)  # 너무 빨리 지나가면 UX가 안 좋아서 살짝 텀을 줌
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_file_path = tmp.name
                if hasattr(image, 'save'):
                    image.save(temp_file_path)
                else:
                    from PIL import Image
                    Image.fromarray(image).save(temp_file_path)
            
            # 2단계: AI 분석
            progress(0.4, desc="🧠 AI가 분위기 분석 중...")
            query_vector, caption, enhanced_caption = self.recommender.get_query_vector(temp_file_path)
            
            if query_vector is None:
                return "❌ 이미지 분석 실패", "", None, ""
            
            analysis_text = f"""
            ### 👁️ AI 시각 분석 결과
            * **📝 설명:** {caption}
            * **✨ 분위기 키워드:** `{enhanced_caption}`
            """

            # 3단계: 음악 매칭
            progress(0.7, desc="🎵 어울리는 음악 찾는 중...")
            time.sleep(0.2)
            
            q_norm = np.linalg.norm(query_vector)
            if q_norm > 0: query_vector = query_vector / q_norm
            similarities = self.vectors @ query_vector
            
            top_indices = np.argsort(similarities)[::-1][:topk]
            results_html = self._format_results(top_indices, similarities)
            
            top_idx = int(top_indices[0])
            top_audio_path = self.metadata[top_idx].get("file_path")
            final_audio_path = top_audio_path if top_audio_path and os.path.exists(top_audio_path) else None
            
            # 4단계: 완료
            progress(1.0, desc="✨ 완료!")
            status_msg = f"✅ 추천 완료! {topk}곡을 찾았습니다."
            
            return status_msg, results_html, final_audio_path, analysis_text
            
        except Exception as e:
            traceback.print_exc()
            return f"❌ 오류 발생: {str(e)}", "", None, ""
        
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except: pass
    
    def _format_results(self, indices, similarities):
        html = "<div style='font-family: sans-serif; padding: 5px;'>"
        for rank, idx in enumerate(indices, 1):
            song = self.metadata[int(idx)]
            score = float(similarities[int(idx)])
            mood = song.get('mood', 'Unknown')
            genre = song.get('genre', 'Unknown')
            title = song.get('title', 'Unknown Title')
            opacity = max(0.6, min(1.0, score + 0.2)) 
            
            html += f"""
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, {opacity}) 0%, rgba(118, 75, 162, {opacity}) 100%);
                border-radius: 12px; padding: 15px; margin-bottom: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <div style='display: flex; justify-content: space-between;'>
                    <div style='font-weight: bold; font-size: 1.1em;'>#{rank} {title}</div>
                    <div>🎵</div>
                </div>
                <div style='font-size: 0.9em; opacity: 0.9; margin-top: 5px;'>
                    🎭 {mood} | 🎸 {genre}
                </div>
                <div style='text-align: right; font-size: 0.8em; opacity: 0.8;'>유사도: {score:.3f}</div>
            </div>"""
        return html + "</div>"

def create_interface():
    try:
        app = MusicRecommenderApp()
    except Exception as e:
        print(f"❌ 앱 초기화 실패: {e}")
        return gr.Blocks()

    with gr.Blocks(theme=gr.themes.Soft(), title="Music Mood Matcher") as demo:
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <h1>🎵 AI Music Mood Matcher</h1>
                <p>이미지를 업로드하면 AI가 분위기를 분석해 딱 맞는 음악을 추천해드립니다.</p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                image_input = gr.Image(label="📸 이미지 업로드", type="pil", height=450)
                with gr.Row():
                    topk_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="추천 개수")
                    submit_btn = gr.Button("✨ 음악 추천받기", variant="primary", scale=2)

            with gr.Column(scale=5):
                analysis_output = gr.Markdown(label="🧠 AI 분석 결과")
                audio_output = gr.Audio(label="🎧 1위 곡 미리듣기", type="filepath")
                caption_output = gr.Textbox(label="상태", show_label=False, text_align="center")
                results_output = gr.HTML(label="추천 리스트")
        
        submit_btn.click(
            fn=lambda img, k: app.recommend(img, int(k)),
            inputs=[image_input, topk_slider],
            outputs=[caption_output, results_output, audio_output, analysis_output]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=False)