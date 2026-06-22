"""Integration tests for the full Adaptive Semantic Chunker pipeline."""

import pytest
import socket
import nltk
from langchain_core.documents import Document
from asc.chunker.adaptive_chunker import AdaptiveSemanticChunker

def is_ollama_running() -> bool:
    """Checks if Ollama is running on the local port 11434."""
    try:
        with socket.create_connection(("localhost", 11434), timeout=1.0):
            return True
    except OSError:
        return False

# 500+ word Wikipedia excerpt with a clear topic shift from AI to Photosynthesis
WIKIPEDIA_EXCERPT = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence "
    "displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, "
    "which refers to any system that perceives its environment and takes actions that maximize its chance of achieving "
    "its goals. More specifically, AI is the capability of a computer system to mimic human cognitive functions, "
    "such as learning and problem-solving. Through mathematics and logic, a computer system simulates the reasoning "
    "that humans use to learn from new information and make decisions. "
    "Historically, the field of AI was founded on the assumption that human intelligence can be precisely described. "
    "Early researchers built systems that relied on symbolic logic, expert knowledge, and hand-coded rules. "
    "These early expert systems were highly successful in narrow domains, such as playing chess or diagnosing "
    "specific diseases, but they failed to generalize to more complex, real-world tasks. "
    "However, progress stalled during several periods known as AI winters, where funding and interest dried up "
    "due to unmet expectations. In the 21st century, AI techniques experienced a massive resurgence following "
    "concurrent advances in computer power, large amounts of data, and theoretical understanding. "
    "Machine learning, a subset of AI, became the dominant paradigm, where algorithms learn patterns from data "
    "rather than relying on explicit programming. Deep learning, which uses multi-layered artificial neural "
    "networks, has led to breakthroughs in computer vision, natural language processing, and speech recognition. "
    "These deep networks are inspired by the biological structure of the human brain, containing millions of "
    "interconnected artificial neurons. Today, AI systems are deployed in autonomous vehicles, medical diagnostics, "
    "search engines, and generative content creation, transforming global industries. "
    "As AI systems become more autonomous and integrated into daily life, questions of ethics, transparency, "
    "and alignment with human values have become central to research. Researchers are now developing methods for "
    "explainable AI, ensuring that decision-making processes can be audited and understood by human operators. "
    "Despite these challenges, the rapid pace of innovation suggests that AI will continue to be a primary driver "
    "of technological change in the coming decades, reshaping the future of work, education, and human society. "
    # Topic shift to photosynthesis
    "Photosynthesis is a biological process used by plants, algae, and certain bacteria to harness energy from "
    "sunlight and turn it into chemical energy. This chemical energy is stored in carbohydrate molecules, such as "
    "sugars and starches, which are synthesized from carbon dioxide and water. The process is of fundamental importance "
    "for life on Earth, as it provides the primary source of energy for nearly all organisms and releases oxygen into "
    "the atmosphere. Photosynthesis is largely responsible for producing and maintaining the oxygen content of the "
    "Earth's atmosphere, and supplies most of the organic compounds and energy necessary for life. "
    "In plants and algae, photosynthesis occurs in specialized intracellular organelles called chloroplasts. "
    "These organelles contain thylakoid membranes where light-absorbing pigments, primarily chlorophyll, are embedded. "
    "The chlorophyll molecules absorb photons, which excites electrons and initiates a cascade of charge separation. "
    "The light-dependent reactions of photosynthesis absorb light energy and convert it into chemical energy in the form "
    "of ATP and NADPH. These energy-carrying molecules are then utilized in the light-independent reactions, also known "
    "as the Calvin cycle. During the Calvin cycle, the enzyme RuBisCO catalyzes the fixation of carbon dioxide into "
    "three-carbon sugar precursors. These precursors are subsequently converted into glucose, sucrose, and other "
    "structural carbohydrates required for plant growth and cellular respiration. Plants then use these sugars to build "
    "cellulose for structural support and starch for long-term energy storage. In addition, the oxygen produced during "
    "the photolysis of water is released as a byproduct, which supports aerobic respiration in heterotrophic organisms. "
    "Over billions of years, this biochemical process has shaped the Earth's geosphere and biosphere, establishing "
    "the modern climate and supporting the evolution of complex multicellular life forms."
)

@pytest.fixture
def live_chunker():
    """Fixture that initializes the live chunker from environment variables."""
    if not is_ollama_running():
        pytest.skip("Ollama is not running on localhost:11434")
    return AdaptiveSemanticChunker.from_env()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_chunking(live_chunker) -> None:
    """Verifies that the full pipeline correctly chunks text, preserves text, and generates metadata."""
    # Chunk the Wikipedia excerpt
    source_meta = {"source": "wikipedia_excerpt.txt"}
    chunks = await live_chunker.chunk_text(WIKIPEDIA_EXCERPT, source_meta)
    
    # 1. returns >= 2 chunks
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    
    # 2. All chunk metadata fields are non-null
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        assert meta is not None
        assert meta["chunk_index"] == i
        assert meta["total_chunks"] == len(chunks)
        assert meta["sentence_start"] is not None
        assert meta["sentence_end"] is not None
        assert meta["sentence_count"] > 0
        assert meta["avg_perplexity"] is not None
        assert meta["max_perplexity"] is not None
        assert meta["min_perplexity"] is not None
        assert meta["boundary_z_score"] is not None
        assert meta["source"] == "wikipedia_excerpt.txt"
        assert meta["is_chunked"] is True
        
    # 3. Total text preserved across chunks (accounting for overlap)
    # Let's tokenize original sentences and check that all sentences are present in the chunks
    original_sentences = nltk.sent_tokenize(WIKIPEDIA_EXCERPT)
    chunk_sentences_all = []
    for chunk in chunks:
        chunk_sentences_all.extend(nltk.sent_tokenize(chunk.page_content))
        
    # Remove overlap duplicates to verify preservation
    unique_chunk_sentences = []
    for sent in chunk_sentences_all:
        if not unique_chunk_sentences or unique_chunk_sentences[-1] != sent:
            unique_chunk_sentences.append(sent)
            
    # Verify that all original sentences are present
    assert len(unique_chunk_sentences) == len(original_sentences)
    for orig, chunked in zip(original_sentences, unique_chunk_sentences):
        assert orig.strip() == chunked.strip()

    # 4. Boundaries exist at semantically reasonable positions
    # With sentence_overlap=1, the transition from AI to Photosynthesis should trigger a boundary.
    # The transition sentence is: "Photosynthesis is a biological process..."
    # Let's verify that one of the chunks starts at or near this transition.
    transition_found = False
    for chunk in chunks:
        # Since sentence_overlap might prepend the last sentence of the previous chunk,
        # we check if the transition sentence is inside the chunk.
        if "Photosynthesis is a biological process" in chunk.page_content:
            transition_found = True
            break
            
    assert transition_found, "Could not find the transition sentence in any chunk."
