import { useState, useRef } from "react";
import { 
  Upload, 
  Download, 
  Plus, 
  Trash2, 
  Edit2, 
  Save, 
  FileText, 
  Sparkles, 
  X, 
  Check, 
  CheckCircle2, 
  AlertCircle 
} from "lucide-react";

export interface WritingSkill {
  id: string;
  name: string;
  description: string;
  content: string;
  isActive: boolean;
  isCustom?: boolean;
}

interface WritingSkillsManagerProps {
  skills: WritingSkill[];
  onChange: (updatedSkills: WritingSkill[]) => void;
}

export function WritingSkillsManager({ skills, onChange }: WritingSkillsManagerProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // State for creating/editing a skill
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingSkillId, setEditingSkillId] = useState<string | null>(null);
  const [formName, setFormName] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [formContent, setFormContent] = useState("");
  
  // Notification state
  const [notify, setNotify] = useState<{ type: 'success' | 'error', message: string } | null>(null);

  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotify({ type, message });
    setTimeout(() => setNotify(null), 3000);
  };

  const handleToggleSkill = (id: string) => {
    const updated = skills.map(s => s.id === id ? { ...s, isActive: !s.isActive } : s);
    onChange(updated);
  };

  const handleOpenCreateForm = () => {
    setEditingSkillId(null);
    setFormName("");
    setFormDescription("");
    setFormContent("");
    setIsFormOpen(true);
  };

  const handleOpenEditForm = (skill: WritingSkill) => {
    setEditingSkillId(skill.id);
    setFormName(skill.name);
    setFormDescription(skill.description);
    setFormContent(skill.content);
    setIsFormOpen(true);
  };

  const handleSaveSkill = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formName.trim() || !formContent.trim()) {
      showNotification('error', 'Vui lòng nhập tên và nội dung kỹ năng!');
      return;
    }

    if (editingSkillId) {
      // Edit existing
      const updated = skills.map(s => 
        s.id === editingSkillId 
          ? { ...s, name: formName, description: formDescription, content: formContent } 
          : s
      );
      onChange(updated);
      showNotification('success', 'Đã cập nhật kỹ năng viết bài!');
    } else {
      // Create new
      const newSkill: WritingSkill = {
        id: `skill_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
        name: formName,
        description: formDescription || "Kỹ năng tùy chỉnh",
        content: formContent,
        isActive: true,
        isCustom: true
      };
      onChange([...skills, newSkill]);
      showNotification('success', 'Đã thêm kỹ năng viết bài mới!');
    }

    setIsFormOpen(false);
    setEditingSkillId(null);
  };

  const handleDeleteSkill = (id: string, name: string) => {
    if (confirm(`Bạn có chắc chắn muốn xóa kỹ năng "${name}"?`)) {
      const updated = skills.filter(s => s.id !== id);
      onChange(updated);
      showNotification('success', 'Đã xóa kỹ năng.');
    }
  };

  const handleImportFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      
      if (file.name.endsWith('.json')) {
        try {
          const parsed = JSON.parse(text);
          const rawList = Array.isArray(parsed) ? parsed : [parsed];
          const importedSkills: WritingSkill[] = [];

          rawList.forEach((item, idx) => {
            if (item && typeof item === 'object') {
              importedSkills.push({
                id: item.id || `skill_imported_${Date.now()}_${idx}_${Math.random().toString(36).substring(2, 5)}`,
                name: item.name || item.title || "Kỹ năng nhập từ file",
                description: item.description || "Nhập từ dữ liệu JSON",
                content: item.content || item.instructions || item.guidelines || "",
                isActive: item.isActive ?? true,
                isCustom: true
              });
            }
          });

          if (importedSkills.length === 0) {
            showNotification('error', 'File JSON không chứa kỹ năng hợp lệ!');
            return;
          }

          // Filter out skills with empty content
          const validSkills = importedSkills.filter(s => s.content.trim() !== "");
          if (validSkills.length === 0) {
            showNotification('error', 'Nội dung hướng dẫn kỹ năng không được để trống!');
            return;
          }

          // Append to state, avoiding duplicate IDs
          const existingIds = new Set(skills.map(s => s.id));
          const filteredImports = validSkills.map(s => {
            if (existingIds.has(s.id)) {
              return { ...s, id: `${s.id}_dup_${Math.random().toString(36).substring(2, 5)}` };
            }
            return s;
          });

          onChange([...skills, ...filteredImports]);
          showNotification('success', `Đã nhập thành công ${filteredImports.length} kỹ năng!`);
        } catch (e) {
          showNotification('error', 'Lỗi phân tích file JSON!');
        }
      } else if (file.name.endsWith('.md') || file.name.endsWith('.markdown') || file.name.endsWith('.txt')) {
        // Handle markdown/text files
        let skillName = file.name.replace(/\.md$|\.markdown$|\.txt$/i, "");
        let desc = "Nhập từ file Markdown";
        let content = text;

        // Try to parse heading 1 if there's one at the top
        const firstLineHeader = text.match(/^#\s+(.+)$/m);
        if (firstLineHeader) {
          skillName = firstLineHeader[1].trim();
          // Remove the first header from content if desired or keep it
          // Let's keep it clean by removing it
          content = text.replace(/^#\s+(.+)$/m, "").trim();
        }

        const newSkill: WritingSkill = {
          id: `skill_md_${Date.now()}_${Math.random().toString(36).substring(2, 5)}`,
          name: skillName,
          description: desc,
          content: content,
          isActive: true,
          isCustom: true
        };

        onChange([...skills, newSkill]);
        showNotification('success', `Đã nhập thành công kỹ năng "${skillName}" từ Markdown!`);
      } else {
        showNotification('error', 'Chỉ hỗ trợ file .json, .md, .markdown hoặc .txt!');
      }
    };

    reader.onerror = () => {
      showNotification('error', 'Không thể đọc file!');
    };

    reader.readAsText(file);
    e.target.value = ""; // Reset file input
  };

  const handleExportSkills = () => {
    if (skills.length === 0) {
      showNotification('error', 'Không có kỹ năng nào để xuất!');
      return;
    }

    try {
      const text = JSON.stringify(skills, null, 2);
      const blob = new Blob([text], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `storycraft_skills_${new Date().toISOString().slice(0,10)}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      showNotification('success', 'Đã xuất danh sách kỹ năng thành công!');
    } catch (e) {
      showNotification('error', 'Không thể xuất file kỹ năng.');
    }
  };

  return (
    <div className="space-y-6">
      {/* Notifications */}
      {notify && (
        <div className={`fixed top-20 right-4 z-50 p-4 rounded-xl shadow-lg border animate-in slide-in-from-right duration-200 flex items-center gap-3 ${
          notify.type === 'success' 
            ? 'bg-emerald-50 border-emerald-100 text-emerald-800' 
            : 'bg-rose-50 border-rose-100 text-rose-800'
        }`}>
          {notify.type === 'success' ? <CheckCircle2 size={18} className="text-emerald-500" /> : <AlertCircle size={18} className="text-rose-500" />}
          <span className="text-sm font-medium">{notify.message}</span>
        </div>
      )}

      {/* Main guideline box */}
      <div className="p-5 bg-indigo-50 border border-indigo-100 rounded-2xl text-sm text-indigo-800 space-y-3">
        <div className="flex items-center gap-2">
          <Sparkles size={18} className="text-indigo-600" />
          <strong className="text-indigo-900 font-bold">Kỹ năng AI (Writing Skills & Prompts)</strong>
        </div>
        <p className="text-xs sm:text-sm text-indigo-700 leading-relaxed">
          Kỹ năng AI cung cấp những định hướng nghệ thuật, phong cách viết chuyên biệt (như <i>Đặc tả tâm lý</i>, <i>Mô tả chiến đấu</i>, hay <i>Show, Don't Tell</i>). 
          Khi được kích hoạt, AI sẽ bám sát các hướng dẫn này để nâng cao chất lượng câu chữ trong từng chương truyện.
        </p>
        <div className="flex flex-wrap gap-3 pt-1">
          <button
            onClick={handleOpenCreateForm}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-bold transition-all shadow-sm"
          >
            <Plus size={14} />
            Thêm kỹ năng thủ công
          </button>
          
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white text-indigo-700 border border-indigo-200 hover:bg-indigo-50 rounded-xl text-xs font-bold transition-all shadow-sm"
          >
            <Upload size={14} />
            Nạp file (.json / .md)
          </button>

          <button
            onClick={handleExportSkills}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white text-indigo-700 border border-indigo-200 hover:bg-indigo-50 rounded-xl text-xs font-bold transition-all shadow-sm"
          >
            <Download size={14} />
            Xuất danh sách (.json)
          </button>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImportFile}
            accept=".json,.md,.markdown,.txt"
            className="hidden"
          />
        </div>
      </div>

      {/* Form Dialog for Add/Edit */}
      {isFormOpen && (
        <div className="p-5 border border-stone-200 bg-stone-50 rounded-2xl space-y-4 animate-in fade-in duration-200">
          <div className="flex justify-between items-center pb-2 border-b border-stone-200">
            <h3 className="font-bold text-stone-800 text-sm flex items-center gap-1.5">
              <Sparkles size={16} className="text-indigo-500" />
              {editingSkillId ? "Chỉnh sửa kỹ năng" : "Tạo kỹ năng viết bài mới"}
            </h3>
            <button 
              type="button"
              onClick={() => setIsFormOpen(false)}
              className="p-1 text-stone-400 hover:text-stone-600 hover:bg-stone-200 rounded-full transition-colors"
            >
              <X size={16} />
            </button>
          </div>
          <form onSubmit={handleSaveSkill} className="space-y-4">
            <div>
              <label className="block text-xs font-bold text-stone-500 uppercase tracking-wider mb-1.5">Tên kỹ năng *</label>
              <input
                type="text"
                required
                value={formName}
                onChange={e => setFormName(e.target.value)}
                placeholder="Ví dụ: Tả cảnh điện ảnh, Độc thoại u ám, Chiến đấu dồn dập..."
                className="w-full px-4 py-2 bg-white border border-stone-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-xs font-bold text-stone-500 uppercase tracking-wider mb-1.5">Mô tả ngắn gọn</label>
              <input
                type="text"
                value={formDescription}
                onChange={e => setFormDescription(e.target.value)}
                placeholder="Ví dụ: Giúp câu chữ tăng sức căng nghệ thuật, nhịp dồn..."
                className="w-full px-4 py-2 bg-white border border-stone-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-xs font-bold text-stone-500 uppercase tracking-wider mb-1.5">Công thức chỉ định cho AI (Prompt Guidelines) *</label>
              <textarea
                required
                rows={4}
                value={formContent}
                onChange={e => setFormContent(e.target.value)}
                placeholder="Nhập chi tiết yêu cầu viết bài. Ví dụ: Hãy tả thật sâu sự buốt lạnh mồ hôi và đôi vai run rẩy. Tránh các từ ngữ trừu tượng."
                className="w-full px-4 py-3 bg-white border border-stone-200 rounded-xl text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setIsFormOpen(false)}
                className="px-4 py-2 bg-white border border-stone-200 hover:bg-stone-100 text-stone-700 rounded-xl text-xs font-bold transition-all"
              >
                Hủy
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-stone-900 hover:bg-stone-800 text-white rounded-xl text-xs font-bold transition-all flex items-center gap-1"
              >
                <Save size={14} />
                Lưu kỹ năng
              </button>
            </div>
          </form>
        </div>
      )}

      {/* List of Skills */}
      <div className="space-y-4">
        <label className="block text-xs font-bold text-stone-500 uppercase tracking-wider">Danh sách kỹ năng hiện có ({skills.length})</label>
        {skills.length === 0 ? (
          <div className="p-8 text-center bg-white border border-dashed border-stone-200 rounded-2xl text-stone-400">
            <Sparkles size={28} className="mx-auto mb-2 opacity-50" />
            <p className="text-sm">Hiện tại chưa có kỹ năng nào. Hãy thêm thủ công hoặc nạp từ file JSON/MD.</p>
          </div>
        ) : (
          <div className="grid gap-3">
            {skills.map((skill) => (
              <div 
                key={skill.id}
                className={`p-4 bg-white rounded-2xl border transition-all relative flex flex-col justify-between md:flex-row md:items-center gap-4 ${
                  skill.isActive 
                    ? "border-indigo-500 ring-1 ring-indigo-50/50 shadow-sm" 
                    : "border-stone-200 hover:border-stone-300"
                }`}
              >
                <div className="space-y-1 max-w-xl">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleToggleSkill(skill.id)}
                      className={`w-5 h-5 rounded-md flex items-center justify-center border transition-all ${
                        skill.isActive 
                          ? "bg-indigo-600 border-indigo-600 text-white" 
                          : "border-stone-300 bg-white hover:border-indigo-500"
                      }`}
                    >
                      {skill.isActive && <Check size={12} strokeWidth={3} />}
                    </button>
                    <h4 className="font-bold text-stone-800 text-sm inline-flex items-center gap-1.5">
                      {skill.name}
                      {skill.isActive && (
                        <span className="px-1.5 py-0.5 bg-indigo-50 text-[10px] text-indigo-600 font-medium rounded-md border border-indigo-100">
                          Đang kích hoạt
                        </span>
                      )}
                    </h4>
                  </div>
                  <p className="text-xs text-stone-400 italic pl-7">{skill.description}</p>
                  <div className="pl-7 pt-1">
                    <details className="group">
                      <summary className="text-[11px] text-indigo-600 font-semibold cursor-pointer select-none outline-none hover:underline">
                        Xem hướng dẫn nhắc lệnh cho AI
                      </summary>
                      <div className="mt-2 p-3 bg-stone-50 border border-stone-100 rounded-xl text-xs font-mono text-stone-600 whitespace-pre-wrap leading-relaxed max-h-40 overflow-y-auto">
                        {skill.content}
                      </div>
                    </details>
                  </div>
                </div>

                <div className="flex items-center gap-2 pl-7 md:pl-0 self-end md:self-center">
                  <button
                    onClick={() => handleOpenEditForm(skill)}
                    className="p-1.5 text-stone-400 hover:text-indigo-600 hover:bg-stone-100 rounded-lg transition-colors"
                    title="Chỉnh sửa kỹ năng"
                  >
                    <Edit2 size={14} />
                  </button>
                  <button
                    onClick={() => handleDeleteSkill(skill.id, skill.name)}
                    className="p-1.5 text-stone-400 hover:text-rose-600 hover:bg-stone-100 rounded-lg transition-colors"
                    title="Xóa kỹ năng"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
